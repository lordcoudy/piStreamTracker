"""Installer CLI: role, network target, argparse."""

import unittest

from pistream.install import (
    InstallError,
    detect_pi_model,
    network_target,
    parse_args,
    resolve_role,
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
