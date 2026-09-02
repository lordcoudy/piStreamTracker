"""piStreamTracker installer CLI (OS network, udev, systemd). No OpenCV/picamera2."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Callable


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


def main(argv: list[str] | None = None) -> int:
    parse_args(argv)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
