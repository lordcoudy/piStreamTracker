"""Bounded per-device USB transactions and release without a real EV3."""

import importlib
import sys
from threading import Event, Lock, Thread
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest


@pytest.fixture
def usb_module():
    # Keep isolated runs camera/USB-library free without changing a module
    # another test has already installed or imported during collection.
    if 'ev3_dc' in sys.modules:
        return importlib.import_module('pistream.ev3_usb')
    with patch.dict(sys.modules, {'ev3_dc': MagicMock()}):
        return importlib.import_module('pistream.ev3_usb')


class Device:
    def __init__(self):
        self.read_timeouts = []
        self.write_timeouts = []
        self.read_error = None
        self.write_error = None

    def read(self, endpoint, length, timeout=0):
        self.read_timeouts.append(timeout)
        if self.read_error:
            raise self.read_error
        return b'reply'

    def write(self, endpoint, data, timeout=0):
        self.write_timeouts.append(timeout)
        if self.write_error:
            raise self.write_error
        return len(data)


class Physical:
    """The installed library's acquire/read/release pattern, including its bug."""

    def __init__(self, device, hosts=None):
        self._device = device
        self._lock = Lock()
        self._host = '00:16:53:40:42:CB'
        self._devices = hosts if hosts is not None else set()
        assert self._host not in self._devices, 'host already connected'
        self._devices.add(self._host)

    def send_direct_cmd(self, ops, **kwargs):
        self._lock.acquire()
        self._device.write(1, ops, 100)
        return self.wait_for_reply(b'counter', _locked=True)

    def wait_for_reply(self, counter, _locked=False, **kwargs):
        if not _locked:
            self._lock.acquire()
        reply = self._device.read(0x81, 1024, 0)
        while reply == b'unrelated':
            reply = self._device.read(0x81, 1024, 0)
        self._lock.release()
        return reply

    def send_system_cmd(self, ops, **kwargs):
        self._lock.acquire()
        self._device.write(1, ops, 100)
        return self._wait_for_system_reply(b'counter', _locked=True)

    def _wait_for_system_reply(self, counter, _locked=False, **kwargs):
        return self.wait_for_reply(counter, _locked=_locked)


class Owner:
    def __init__(self, physical):
        self._physical_ev3 = physical

    def send_direct_cmd(self, ops, **kwargs):
        return self._physical_ev3.send_direct_cmd(ops, **kwargs)


@pytest.fixture
def wrapper(usb_module, monkeypatch):
    devices, hosts, owners = [], set(), []
    def connect(**kwargs):
        device = Device()
        owner = Owner(Physical(device, hosts))
        devices.append(device)
        owners.append(owner)
        return owner
    def motor(port, **kwargs):
        kwargs['ev3_obj'].send_direct_cmd(b'motor introspection')
        return Mock(position=0)
    def jukebox(**kwargs):
        kwargs['ev3_obj'].send_direct_cmd(b'jukebox introspection')
        return Mock()
    library = SimpleNamespace(USB='USB', EV3=Mock(side_effect=connect), Motor=motor, Jukebox=jukebox)
    monkeypatch.setattr(usb_module, 'ev3', library)
    dispose = Mock()
    monkeypatch.setattr(usb_module, '_dispose_usb_device', dispose)
    return usb_module, library, devices, owners, hosts, dispose


@pytest.mark.parametrize('operation', ['send_direct_cmd', 'send_system_cmd', 'wait_for_reply'])
def test_usb_infinite_read_becomes_finite_for_all_shared_commands(usb_module, operation):
    device = Device()
    physical = Physical(device)
    usb_module._BoundedUSBTransport(physical, timeout=0.1)
    assert getattr(physical, operation)(b'command') == b'reply'
    assert device.read_timeouts and all(0 < timeout <= 100 for timeout in device.read_timeouts)


@pytest.mark.parametrize('failure', ['read', 'write'])
def test_usb_error_releases_library_lock_before_next_command(usb_module, failure):
    device = Device()
    physical = Physical(device)
    usb_module._BoundedUSBTransport(physical, timeout=0.1)
    setattr(device, failure + '_error', OSError('USB failed'))
    with pytest.raises(OSError, match='USB failed'):
        physical.send_direct_cmd(b'first')
    setattr(device, failure + '_error', None)
    assert physical.send_direct_cmd(b'next') == b'reply'


def test_silent_reply_timeout_releases_lock_for_stop_and_close(usb_module):
    device = Device()
    physical = Physical(device)
    transport = usb_module._BoundedUSBTransport(physical, timeout=0.1)
    device.read_error = TimeoutError('no EV3 reply')
    with pytest.raises(TimeoutError):
        physical.send_direct_cmd(b'keep alive')
    device.read_error = None
    assert physical.send_direct_cmd(b'stop') == b'reply'
    assert transport._depth() == 0


def test_stream_of_unrelated_replies_cannot_extend_absolute_deadline(usb_module, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(usb_module.time, 'monotonic', lambda: clock[0])
    device = Device()
    def unrelated(endpoint, length, timeout):
        device.read_timeouts.append(timeout)
        clock[0] += 0.04
        return b'unrelated'
    device.read = unrelated
    physical = Physical(device)
    transport = usb_module._BoundedUSBTransport(physical, timeout=0.1)
    with pytest.raises(TimeoutError):
        physical.send_direct_cmd(b'command')
    assert 1 <= len(device.read_timeouts) <= 3
    assert device.read_timeouts[-1] < device.read_timeouts[0]
    assert transport._depth() == 0


def test_lock_wait_is_bounded_and_never_releases_another_threads_ownership(usb_module):
    physical = Physical(Device())
    usb_module._BoundedUSBTransport(physical, timeout=0.02)
    held, release = Event(), Event()
    def holder():
        physical._lock.acquire()
        held.set()
        release.wait(1)
        physical._lock.release()
    thread = Thread(target=holder)
    thread.start()
    try:
        assert held.wait(1)
        with pytest.raises(TimeoutError):
            physical.send_direct_cmd(b'command')
    finally:
        release.set()
        thread.join(1)
    assert not thread.is_alive()
    assert physical.send_direct_cmd(b'after release') == b'reply'


def test_motor_and_led_initialization_use_attached_bounded_transport(wrapper):
    module, library, devices, owners, hosts, dispose = wrapper
    brick = module.EV3_USB()
    brick.Motor('a')
    brick.Led('green', 'pulse')
    assert len(devices[0].read_timeouts) == 2
    assert all(0 < timeout <= 1000 for timeout in devices[0].read_timeouts)
    brick.close()


def test_explicit_close_releases_usb_host_for_next_owner_and_is_idempotent(wrapper):
    module, library, devices, owners, hosts, dispose = wrapper
    brick = module.EV3_USB()
    old_physical = owners[0]._physical_ev3
    brick.close()
    assert hosts == set()
    assert owners[0]._physical_ev3 is None
    dispose.assert_called_once_with(devices[0])
    replacement = module.EV3_USB()
    assert hosts == {old_physical._host}
    brick.close()  # Closing the old owner must not unregister its replacement.
    assert hosts == {old_physical._host}
    with pytest.raises(RuntimeError, match='closed'):
        old_physical.send_direct_cmd(b'command')
    replacement.close()
    assert dispose.call_count == 2


def test_disposal_failure_retains_owner_for_explicit_retry(wrapper):
    module, library, devices, owners, hosts, dispose = wrapper
    brick = module.EV3_USB()
    dispose.side_effect = OSError('USB release failed')
    with pytest.raises(OSError, match='USB release failed'):
        brick.close()
    assert brick._ev3 is owners[0]
    assert hosts
    dispose.side_effect = None
    brick.close()
    assert hosts == set()


def test_close_waits_for_its_inflight_transaction_before_disposing(wrapper):
    module, library, devices, owners, hosts, dispose = wrapper
    brick = module.EV3_USB()
    entered, release = Event(), Event()
    errors = []
    def blocking_read(endpoint, length, timeout):
        entered.set()
        assert release.wait(1)
        return b'reply'
    devices[0].read = blocking_read
    def invoke(action):
        try:
            action()
        except Exception as exc:
            errors.append(exc)
    caller = Thread(target=invoke, args=(lambda: owners[0].send_direct_cmd(b'command'),))
    closer = Thread(target=invoke, args=(brick.close,))
    caller.start()
    assert entered.wait(1)
    closer.start()
    try:
        dispose.assert_not_called()
    finally:
        release.set()
        caller.join(1)
        closer.join(1)
    assert not caller.is_alive() and not closer.is_alive()
    assert errors == []
    dispose.assert_called_once_with(devices[0])
    assert hosts == set()


def test_adapter_does_not_patch_other_usb_instances(usb_module):
    first, second = Physical(Device()), Physical(Device())
    raw_second = second._device
    usb_module._BoundedUSBTransport(first)
    assert second._device is raw_second
    assert second.send_direct_cmd.__func__ is Physical.send_direct_cmd
    assert type(second._lock) is type(Lock())


def test_adapter_setup_failure_disposes_already_opened_usb_owner(wrapper, monkeypatch):
    module, library, devices, owners, hosts, dispose = wrapper
    monkeypatch.setattr(module, '_BoundedUSBTransport', Mock(side_effect=RuntimeError('adapter failed')))
    with pytest.raises(RuntimeError, match='adapter failed'):
        module.EV3_USB()
    dispose.assert_called_once_with(devices[0])
    assert hosts == set()
    assert owners[0]._physical_ev3 is None


def test_stop_does_not_query_busy_before_issuing_stop(usb_module):
    class Motor:
        @property
        def busy(self):
            raise AssertionError('Stop must not require a reply before sending its opcode')

        stop = Mock()
    motor = Motor()
    usb_module.Motor(motor).stop()
    motor.stop.assert_called_once_with()
