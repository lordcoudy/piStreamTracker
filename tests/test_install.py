"""Installer CLI: role, network target, argparse."""

import unittest

from pistream.install import (
    InstallError,
    detect_network_backend,
    detect_pi_model,
    network_target,
    parse_args,
    parse_nmcli_connections,
    plan_networkd_actions,
    plan_nm_actions,
    render_networkd_file,
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

