"""Installer CLI: role, network target, argparse."""

import unittest
from io import StringIO
from pathlib import Path

from pistream.install import (
    InstallError,
    RunAction,
    WriteAction,
    cmd_network,
    detect_network_backend,
    detect_pi_model,
    execute_actions,
    network_target,
    parse_args,
    parse_nmcli_connections,
    plan_networkd_actions,
    plan_nm_actions,
    plan_service_actions,
    plan_udev_actions,
    render_networkd_file,
    render_udev_rule,
    render_unit_file,
    resolve_role,
    ssh_on_interface,
)


class DetectPiModelTests(unittest.TestCase):
    def test_pi5(self):
        self.assertEqual(detect_pi_model('Raspberry Pi 5 Model B Rev 1.0'), 'pi5')

    def test_pi3(self):
        self.assertEqual(detect_pi_model('Raspberry Pi 3 Model B Plus Rev 1.3'), 'pi3')

    def test_unknown_none(self):
        self.assertEqual(detect_pi_model(None), 'unknown')


class ResolveRoleTests(unittest.TestCase):
    def test_explicit_camera(self):
        self.assertEqual(resolve_role('camera', 'pi5'), 'camera')

    def test_auto_pi3_is_camera(self):
        self.assertEqual(resolve_role('auto', 'pi3'), 'camera')

    def test_auto_pi5_is_tracker(self):
        self.assertEqual(resolve_role('auto', 'pi5'), 'tracker')

    def test_auto_unknown_raises(self):
        with self.assertRaises(InstallError):
            resolve_role('auto', 'unknown')


class NetworkTargetTests(unittest.TestCase):
    def test_reads_config_and_role_ip(self):
        cfg = {
            'network': {
                'camera_ip': '192.168.100.1',
                'tracker_ip': '192.168.100.2',
                'interface': 'eth0',
                'subnet': '24',
            }
        }
        self.assertEqual(
            network_target(cfg, 'camera'),
            ('eth0', '192.168.100.1', 24),
        )
        self.assertEqual(
            network_target(cfg, 'tracker'),
            ('eth0', '192.168.100.2', 24),
        )

    def test_interface_override(self):
        cfg = {
            'network': {
                'camera_ip': '10.0.0.1',
                'tracker_ip': '10.0.0.2',
                'interface': 'eth0',
                'subnet': 24,
            }
        }
        self.assertEqual(
            network_target(cfg, 'camera', interface_override='end0'),
            ('end0', '10.0.0.1', 24),
        )

    def test_defaults_when_keys_missing(self):
        cfg = {'network': {'camera_ip': '192.168.100.1', 'tracker_ip': '192.168.100.2'}}
        iface, ip, prefix = network_target(cfg, 'tracker')
        self.assertEqual(iface, 'eth0')
        self.assertEqual(ip, '192.168.100.2')
        self.assertEqual(prefix, 24)


class ParseArgsTests(unittest.TestCase):
    def test_install_camera_skip_flags(self):
        args = parse_args([
            '--dry-run', 'install', '--role', 'camera',
            '--skip-network', '--skip-service',
        ])
        self.assertEqual(args.command, 'install')
        self.assertEqual(args.role, 'camera')
        self.assertTrue(args.dry_run)
        self.assertTrue(args.skip_network)
        self.assertTrue(args.skip_service)
        self.assertFalse(args.skip_udev)

    def test_network_subcommand(self):
        args = parse_args(['network', '--role', 'tracker', '--interface', 'end0'])
        self.assertEqual(args.command, 'network')
        self.assertEqual(args.role, 'tracker')
        self.assertEqual(args.interface, 'end0')

    def test_missing_command_exits(self):
        with self.assertRaises(SystemExit):
            parse_args([])


class NmcliParseTests(unittest.TestCase):
    def test_splits_name_device_type(self):
        text = (
            'Wired connection 1:eth0:802-3-ethernet\n'
            'Wi-Fi:wlan0:802-11-wireless\n'
        )
        rows = parse_nmcli_connections(text)
        self.assertEqual(
            rows,
            [
                ('Wired connection 1', 'eth0', '802-3-ethernet'),
                ('Wi-Fi', 'wlan0', '802-11-wireless'),
            ],
        )


class PlanNmActionsTests(unittest.TestCase):
    def test_modifies_existing_ethernet_never_default(self):
        actions = plan_nm_actions(
            'eth0',
            '192.168.100.1',
            24,
            [
                ('Wired connection 1', 'eth0', '802-3-ethernet'),
                ('Wi-Fi', 'wlan0', '802-11-wireless'),
            ],
        )
        joined = [' '.join(a.argv) for a in actions]
        self.assertTrue(any('connection modify' in j and 'Wired connection 1' in j for j in joined))
        self.assertTrue(any('ipv4.never-default yes' in j for j in joined))
        self.assertTrue(any('192.168.100.1/24' in j for j in joined))
        self.assertTrue(any('connection up' in j and 'Wired connection 1' in j for j in joined))
        self.assertFalse(any('Wi-Fi' in j for j in joined))

    def test_adds_profile_when_no_ethernet(self):
        actions = plan_nm_actions('eth0', '192.168.100.2', 24, [
            ('Wi-Fi', 'wlan0', '802-11-wireless'),
        ])
        joined = [' '.join(a.argv) for a in actions]
        self.assertTrue(any('connection add' in j and 'pistream-eth' in j for j in joined))
        self.assertTrue(any('ifname eth0' in j for j in joined))
        self.assertTrue(any('connection up' in j and 'pistream-eth' in j for j in joined))


class NetworkdTests(unittest.TestCase):
    def test_unit_file_contents(self):
        body = render_networkd_file('eth0', '192.168.100.2', 24)
        self.assertIn('Name=eth0', body)
        self.assertIn('Address=192.168.100.2/24', body)

    def test_plan_writes_and_restarts(self):
        actions = plan_networkd_actions('eth0', '192.168.100.2', 24)
        writes = [a for a in actions if getattr(a, 'path', None)]
        runs = [a for a in actions if getattr(a, 'argv', None)]
        self.assertEqual(writes[0].path, '/etc/systemd/network/10-pistream.network')
        self.assertIn('systemctl', runs[0].argv)
        self.assertIn('restart', runs[0].argv)
        self.assertIn('systemd-networkd', runs[0].argv)


class BackendTests(unittest.TestCase):
    def test_prefers_networkmanager(self):
        self.assertEqual(
            detect_network_backend(lambda n: n == 'NetworkManager'),
            'networkmanager',
        )

    def test_falls_back_to_networkd(self):
        self.assertEqual(
            detect_network_backend(lambda n: n == 'systemd-networkd'),
            'networkd',
        )

    def test_neither_raises(self):
        with self.assertRaises(InstallError):
            detect_network_backend(lambda n: False)


class SshGuardTests(unittest.TestCase):
    def test_server_ip_on_iface(self):
        self.assertTrue(
            ssh_on_interface('10.0.0.8 12345 192.168.100.1 22', {'192.168.100.1'})
        )

    def test_unrelated_ssh(self):
        self.assertFalse(
            ssh_on_interface('10.0.0.8 12345 192.168.1.20 22', {'192.168.100.1'})
        )

    def test_missing_env(self):
        self.assertFalse(ssh_on_interface(None, {'192.168.100.1'}))


class UnitFileTests(unittest.TestCase):
    def test_camera_unit(self):
        body = render_unit_file(
            description='piStreamTracker Camera Server',
            user='pi',
            work_dir='/opt/piStreamTracker',
            python='/opt/piStreamTracker/venv/bin/python',
            script='/opt/piStreamTracker/camera.py',
        )
        self.assertIn('User=pi', body)
        self.assertIn('WorkingDirectory=/opt/piStreamTracker', body)
        self.assertIn(
            'ExecStart=/opt/piStreamTracker/venv/bin/python /opt/piStreamTracker/camera.py',
            body,
        )
        self.assertIn('After=network-online.target', body)

    def test_service_plan_enable(self):
        actions = plan_service_actions(
            role='tracker',
            user='pi',
            work_dir='/repo',
            python='/repo/venv/bin/python',
            enable=True,
        )
        writes = [a for a in actions if isinstance(a, WriteAction)]
        self.assertEqual(writes[0].path, '/etc/systemd/system/pitracker.service')
        self.assertIn('web.py', writes[0].content)
        joined = [' '.join(a.argv) for a in actions if isinstance(a, RunAction)]
        self.assertTrue(any('daemon-reload' in j for j in joined))
        self.assertTrue(any('enable' in j and '--now' in j for j in joined))


class UdevTests(unittest.TestCase):
    def test_rule_vendor(self):
        rule = render_udev_rule()
        self.assertIn('idVendor}=="0694"', rule)
        self.assertIn('GROUP="plugdev"', rule)

    def test_plan_includes_usermod(self):
        actions = plan_udev_actions('milord')
        joined = [' '.join(a.argv) for a in actions if isinstance(a, RunAction)]
        self.assertTrue(any('usermod' in j and 'milord' in j for j in joined))


class ExecuteActionsTests(unittest.TestCase):
    def test_dry_run_does_not_run(self):
        calls = []
        writes = []
        out = StringIO()
        execute_actions(
            [RunAction(('nmcli', 'connection', 'up', 'x')), WriteAction('/tmp/a', 'hi')],
            dry_run=True,
            run=lambda *a, **k: calls.append((a, k)),
            write_file=lambda p, c: writes.append((p, c)),
            stdout=out,
        )
        self.assertEqual(calls, [])
        self.assertEqual(writes, [])
        self.assertIn('nmcli', out.getvalue())
        self.assertIn('/tmp/a', out.getvalue())

    def test_run_prefixes_sudo(self):
        calls = []
        execute_actions(
            [RunAction(('nmcli', 'connection', 'up', 'x'))],
            dry_run=False,
            run=lambda argv, **k: calls.append(argv),
            write_file=lambda p, c: None,
            stdout=StringIO(),
        )
        self.assertEqual(calls[0][:2], ('sudo', 'nmcli'))


class InstallModuleTests(unittest.TestCase):
    def test_install_module_has_no_heavy_imports(self):
        source = Path(__file__).resolve().parents[1] / 'pistream' / 'install.py'
        text = source.read_text()
        self.assertNotIn('cv2', text)
        self.assertNotIn('picamera2', text)
        self.assertNotIn('flask', text.lower())


class CmdNetworkTests(unittest.TestCase):
    def test_non_linux_skips(self):
        out = StringIO()
        args = parse_args(['--dry-run', 'network', '--role', 'camera'])
        rc = cmd_network(args, stdout=out, platform='darwin')
        self.assertEqual(rc, 0)
        self.assertIn('not Linux', out.getvalue())

    def test_linux_nm_dry_run_uses_config_ip(self):
        out = StringIO()
        calls = []
        args = parse_args(['--dry-run', 'network', '--role', 'camera'])
        rc = cmd_network(
            args,
            load_cfg=lambda: {
                'network': {
                    'camera_ip': '192.168.100.1',
                    'tracker_ip': '192.168.100.2',
                    'interface': 'eth0',
                    'subnet': 24,
                }
            },
            is_active=lambda name: name == 'NetworkManager',
            list_connections=lambda: [
                ('Wired connection 1', 'eth0', '802-3-ethernet'),
            ],
            run=lambda *a, **k: calls.append((a, k)),
            write_file=lambda p, c: None,
            stdout=out,
            platform='linux',
            model_text='Raspberry Pi 3 Model B Plus',
        )
        self.assertEqual(rc, 0)
        self.assertEqual(calls, [])
        self.assertIn('192.168.100.1/24', out.getvalue())
        self.assertIn('never-default', out.getvalue())



