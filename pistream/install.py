"""piStreamTracker installer CLI (OS network, udev, systemd)."""

from __future__ import annotations

import argparse
import getpass
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TextIO


class InstallError(Exception):
    """User-facing installer failure."""


def detect_pi_model(model_text: str | None) -> str:
    if not model_text:
        return 'unknown'
    if 'Pi 5' in model_text:
        return 'pi5'
    if 'Pi 4' in model_text:
        return 'pi4'
    if 'Pi 3' in model_text:
        return 'pi3'
    return 'unknown'


def resolve_role(requested: str, model: str) -> str:
    if requested in ('camera', 'tracker'):
        return requested
    if requested == 'auto':
        if model == 'pi3':
            return 'camera'
        if model == 'pi5':
            return 'tracker'
        raise InstallError(
            f'Cannot auto-detect role from Pi model {model!r}; pass --role camera or --role tracker'
        )
    raise InstallError(f'Unknown role: {requested}')


def network_target(
    config: dict[str, Any],
    role: str,
    interface_override: str | None = None,
) -> tuple[str, str, int]:
    net = config.get('network') or {}
    key = 'camera_ip' if role == 'camera' else 'tracker_ip'
    ip = str(net.get(key) or '').strip()
    if not ip:
        raise InstallError(f'network.{key} is missing')
    iface = (interface_override or net.get('interface') or 'eth0')
    iface = str(iface).strip()
    if not iface:
        raise InstallError('network.interface is empty')
    raw_prefix = net.get('subnet', 24)
    try:
        prefix = int(str(raw_prefix).strip())
    except (TypeError, ValueError) as exc:
        raise InstallError(f'network.subnet must be an integer prefix: {raw_prefix!r}') from exc
    if not 1 <= prefix <= 32:
        raise InstallError('network.subnet must be between 1 and 32')
    return iface, ip, prefix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='pistream-install',
        description='Install piStreamTracker network, udev, and systemd units',
    )
    parser.add_argument('--config', default='config.yaml', help='Config file')
    parser.add_argument('--dry-run', action='store_true', help='Print actions without sudo')
    sub = parser.add_subparsers(dest='command', required=True)

    inst = sub.add_parser('install', help='Full automatic install (network + service + udev)')
    inst.add_argument('--role', choices=('camera', 'tracker', 'auto'), default='auto')
    inst.add_argument('--skip-network', action='store_true')
    inst.add_argument('--skip-service', action='store_true')
    inst.add_argument('--skip-udev', action='store_true')
    inst.add_argument('--interface', default=None, help='Override network.interface')

    net = sub.add_parser('network', help='Configure ethernet IPv4 from config.yaml')
    net.add_argument('--role', choices=('camera', 'tracker', 'auto'), default='auto')
    net.add_argument('--interface', default=None)

    svc = sub.add_parser('service', help='Write and enable pitracker.service')
    svc.add_argument('--role', choices=('camera', 'tracker', 'auto'), default='auto')
    svc.add_argument('--no-enable', action='store_true', help='Write unit but do not enable')

    sub.add_parser('udev', help='Install EV3 USB udev rule')
    sub.add_parser('status', help='Show network, service, and udev state')
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


@dataclass(frozen=True)
class RunAction:
    argv: tuple[str, ...]
    description: str = ''


@dataclass(frozen=True)
class WriteAction:
    path: str
    content: str
    description: str = ''


Action = RunAction | WriteAction

_WIFI_TYPES = {'wifi', '802-11-wireless'}
_ETH_TYPES = {'ethernet', '802-3-ethernet'}


def parse_nmcli_connections(text: str) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        name, device, conn_type = line.rsplit(':', 2)
        rows.append((name, device, conn_type))
    return rows


def _is_wifi(conn_type: str) -> bool:
    return conn_type in _WIFI_TYPES or 'wireless' in conn_type


def _is_ethernet(conn_type: str) -> bool:
    return conn_type in _ETH_TYPES or 'ethernet' in conn_type


def plan_nm_actions(
    interface: str,
    address: str,
    prefix: int,
    connections: list[tuple[str, str, str]],
) -> list[RunAction]:
    cidr = f'{address}/{prefix}'
    chosen = None
    ethernet: list[tuple[str, str, str]] = []
    for name, device, conn_type in connections:
        if _is_wifi(conn_type):
            continue
        if _is_ethernet(conn_type):
            ethernet.append((name, device, conn_type))
    for name, device, _conn_type in ethernet:
        if device == interface:
            chosen = name
            break
    if chosen is None:
        for name, device, _conn_type in ethernet:
            if not device:
                chosen = name
                break
    actions: list[RunAction] = []
    if chosen is None:
        actions.append(RunAction(
            argv=(
                'nmcli', 'connection', 'add', 'type', 'ethernet',
                'con-name', 'pistream-eth', 'ifname', interface,
                'ipv4.method', 'manual', 'ipv4.addresses', cidr,
                'ipv4.never-default', 'yes',
            ),
            description=f'Create ethernet profile pistream-eth on {interface}',
        ))
        chosen = 'pistream-eth'
    else:
        actions.append(RunAction(
            argv=(
                'nmcli', 'connection', 'modify', chosen,
                'ipv4.method', 'manual',
                'ipv4.addresses', cidr,
                'ipv4.gateway', '',
                'ipv4.never-default', 'yes',
            ),
            description=f'Set static {cidr} on {chosen} ({interface})',
        ))
    actions.append(RunAction(
        argv=('nmcli', 'connection', 'up', chosen),
        description=f'Bring up {chosen}',
    ))
    return actions


def render_networkd_file(interface: str, address: str, prefix: int) -> str:
    return (
        '[Match]\n'
        f'Name={interface}\n'
        '\n'
        '[Network]\n'
        f'Address={address}/{prefix}\n'
    )


def plan_networkd_actions(interface: str, address: str, prefix: int) -> list[Action]:
    return [
        WriteAction(
            path='/etc/systemd/network/10-pistream.network',
            content=render_networkd_file(interface, address, prefix),
            description='Write systemd-networkd config',
        ),
        RunAction(
            argv=('systemctl', 'restart', 'systemd-networkd'),
            description='Restart systemd-networkd',
        ),
    ]


def detect_network_backend(is_active: Callable[[str], bool]) -> str:
    if is_active('NetworkManager'):
        return 'networkmanager'
    if is_active('systemd-networkd'):
        return 'networkd'
    raise InstallError(
        'Neither NetworkManager nor systemd-networkd is active. '
        'Install network-manager or pass --skip-network'
    )


def ssh_on_interface(ssh_connection: str | None, interface_ips: set[str]) -> bool:
    if not ssh_connection:
        return False
    parts = ssh_connection.split()
    if len(parts) < 3:
        return False
    return parts[2] in interface_ips


def render_unit_file(
    *,
    description: str,
    user: str,
    work_dir: str,
    python: str,
    script: str,
) -> str:
    return (
        '[Unit]\n'
        f'Description={description}\n'
        'Wants=network-online.target\n'
        'After=network-online.target\n'
        '\n'
        '[Service]\n'
        'Type=simple\n'
        f'User={user}\n'
        f'WorkingDirectory={work_dir}\n'
        'Environment=OPENBLAS_NUM_THREADS=2\n'
        'Environment=OMP_NUM_THREADS=2\n'
        'Environment=MKL_NUM_THREADS=2\n'
        f'ExecStart={python} {script}\n'
        'Restart=always\n'
        'RestartSec=5\n'
        '\n'
        '[Install]\n'
        'WantedBy=multi-user.target\n'
    )


def render_udev_rule() -> str:
    return 'SUBSYSTEM=="usb", ATTR{idVendor}=="0694", MODE="0660", GROUP="plugdev"\n'


def plan_service_actions(
    *,
    role: str,
    user: str,
    work_dir: str,
    python: str,
    enable: bool,
) -> list[Action]:
    script_name = 'camera.py' if role == 'camera' else 'web.py'
    description = (
        'piStreamTracker Camera Server'
        if role == 'camera'
        else 'piStreamTracker Web Interface'
    )
    script = f'{work_dir}/{script_name}'
    actions: list[Action] = [
        WriteAction(
            path='/etc/systemd/system/pitracker.service',
            content=render_unit_file(
                description=description,
                user=user,
                work_dir=work_dir,
                python=python,
                script=script,
            ),
            description='Write pitracker.service',
        ),
        RunAction(('systemctl', 'daemon-reload'), 'Reload systemd'),
    ]
    if enable:
        actions.append(
            RunAction(
                ('systemctl', 'enable', '--now', 'pitracker'),
                'Enable and start pitracker',
            )
        )
    return actions


def plan_udev_actions(username: str) -> list[Action]:
    return [
        WriteAction(
            path='/etc/udev/rules.d/99-ev3.rules',
            content=render_udev_rule(),
            description='Install EV3 udev rule',
        ),
        RunAction(('udevadm', 'control', '--reload-rules'), 'Reload udev rules'),
        RunAction(('usermod', '-aG', 'plugdev,video', username), 'Add user to plugdev,video'),
    ]


def execute_actions(
    actions: list[Action],
    *,
    dry_run: bool = False,
    run: Callable[..., Any] | None = None,
    write_file: Callable[[str, str], None] | None = None,
    stdout: TextIO | None = None,
) -> None:
    if stdout is None:
        stdout = sys.stdout
    if run is None:
        run = subprocess.run
    if write_file is None:
        write_file = lambda path, content: _sudo_write(path, content, run=run)
    for action in actions:
        if isinstance(action, WriteAction):
            if dry_run:
                print(f'# write {action.path}', file=stdout)
                print(action.content, file=stdout)
            else:
                write_file(action.path, action.content)
        elif isinstance(action, RunAction):
            cmd = ('sudo',) + action.argv
            if dry_run:
                print('+ ' + ' '.join(cmd), file=stdout)
            else:
                run(cmd, check=True)


def _sudo_write(path: str, content: str, run: Callable[..., Any] = subprocess.run) -> None:
    run(
        ('sudo', 'tee', path),
        input=content.encode(),
        check=True,
        stdout=subprocess.DEVNULL,
    )


def read_pi_model() -> str | None:
    try:
        return Path('/proc/device-tree/model').read_text().rstrip('\x00')
    except OSError:
        return None


def resolve_python(work_dir: Path) -> str:
    for candidate in (
        work_dir / 'venv' / 'bin' / 'python',
        work_dir / '.venv' / 'bin' / 'python',
    ):
        if os.access(candidate, os.X_OK):
            return str(candidate)
    return sys.executable


def _systemctl_is_active(name: str) -> bool:
    try:
        result = subprocess.run(
            ('systemctl', 'is-active', '--quiet', name),
            check=False,
        )
        return result.returncode == 0
    except OSError:
        return False


def _list_nm_connections() -> list[tuple[str, str, str]]:
    result = subprocess.run(
        ('nmcli', '-g', 'NAME,DEVICE,TYPE', 'connection', 'show'),
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_nmcli_connections(result.stdout)


def _load_config(path: str) -> dict[str, Any]:
    from pistream.config import load_config
    return load_config(path)


def _resolved_role(args: argparse.Namespace, model_text: str | None) -> str:
    if model_text is None:
        model_text = read_pi_model()
    return resolve_role(args.role, detect_pi_model(model_text))


def cmd_network(
    args: argparse.Namespace,
    *,
    load_cfg: Callable[[], dict[str, Any]] | None = None,
    is_active: Callable[[str], bool] | None = None,
    list_connections: Callable[[], list[tuple[str, str, str]]] | None = None,
    run: Callable[..., Any] | None = None,
    write_file: Callable[[str, str], None] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    platform: str | None = None,
    model_text: str | None = None,
    **_kwargs: Any,
) -> int:
    if stdout is None:
        stdout = sys.stdout
    if stderr is None:
        stderr = sys.stderr
    plat = sys.platform if platform is None else platform
    if not plat.startswith('linux'):
        print('Skipping network setup (not Linux).', file=stdout)
        return 0
    config = load_cfg() if load_cfg is not None else _load_config(args.config)
    role = _resolved_role(args, model_text)
    iface, ip, prefix = network_target(config, role, getattr(args, 'interface', None))
    backend = detect_network_backend(is_active or _systemctl_is_active)
    if backend == 'networkmanager':
        connections = (list_connections or _list_nm_connections)()
        actions: list[Action] = list(plan_nm_actions(iface, ip, prefix, connections))
    else:
        actions = plan_networkd_actions(iface, ip, prefix)
    if ssh_on_interface(os.environ.get('SSH_CONNECTION'), {ip}):
        print(
            f'Warning: SSH appears to use {ip} on {iface}; applying static IP anyway. '
            'Pass --skip-network if this session is on that interface.',
            file=stderr,
        )
    execute_actions(
        actions,
        dry_run=bool(getattr(args, 'dry_run', False)),
        run=run,
        write_file=write_file,
        stdout=stdout,
    )
    return 0


def cmd_service(
    args: argparse.Namespace,
    *,
    run: Callable[..., Any] | None = None,
    write_file: Callable[[str, str], None] | None = None,
    stdout: TextIO | None = None,
    platform: str | None = None,
    model_text: str | None = None,
    work_dir: Path | None = None,
    user: str | None = None,
    python: str | None = None,
    **_kwargs: Any,
) -> int:
    if stdout is None:
        stdout = sys.stdout
    plat = sys.platform if platform is None else platform
    if not plat.startswith('linux'):
        print('Skipping systemd service (not Linux).', file=stdout)
        return 0
    from pistream.config import project_root
    root = work_dir if work_dir is not None else project_root()
    role = _resolved_role(args, model_text)
    actions = plan_service_actions(
        role=role,
        user=user if user is not None else getpass.getuser(),
        work_dir=str(root),
        python=python if python is not None else resolve_python(Path(root)),
        enable=not bool(getattr(args, 'no_enable', False)),
    )
    execute_actions(
        actions,
        dry_run=bool(getattr(args, 'dry_run', False)),
        run=run,
        write_file=write_file,
        stdout=stdout,
    )
    return 0


def cmd_udev(
    args: argparse.Namespace,
    *,
    run: Callable[..., Any] | None = None,
    write_file: Callable[[str, str], None] | None = None,
    stdout: TextIO | None = None,
    platform: str | None = None,
    user: str | None = None,
    **_kwargs: Any,
) -> int:
    if stdout is None:
        stdout = sys.stdout
    plat = sys.platform if platform is None else platform
    if not plat.startswith('linux'):
        print('Skipping EV3 udev rule (not Linux).', file=stdout)
        return 0
    actions = plan_udev_actions(user if user is not None else getpass.getuser())
    execute_actions(
        actions,
        dry_run=bool(getattr(args, 'dry_run', False)),
        run=run,
        write_file=write_file,
        stdout=stdout,
    )
    return 0


def cmd_install(
    args: argparse.Namespace,
    **kwargs: Any,
) -> int:
    model_text = kwargs.get('model_text')
    role = _resolved_role(args, model_text)
    args.role = role
    if not getattr(args, 'skip_network', False):
        rc = cmd_network(args, **kwargs)
        if rc:
            return rc
    if role == 'tracker' and not getattr(args, 'skip_udev', False):
        rc = cmd_udev(args, **kwargs)
        if rc:
            return rc
    if not getattr(args, 'skip_service', False):
        rc = cmd_service(args, **kwargs)
        if rc:
            return rc
    return 0


def cmd_status(args: argparse.Namespace, **_kwargs: Any) -> int:
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if args.command == 'install':
            return cmd_install(args)
        if args.command == 'network':
            return cmd_network(args)
        if args.command == 'service':
            return cmd_service(args)
        if args.command == 'udev':
            return cmd_udev(args)
        if args.command == 'status':
            return cmd_status(args)
        raise InstallError(f'Unknown command: {args.command}')
    except InstallError as exc:
        print(exc, file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

