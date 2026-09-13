import ctypes as C
import math
import time

import pytest

from curve_wipe.sensor import (KunweiSensor, SensorConfig, SensorError,
                               StaleSampleError)


class FakeFunction:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *args):
        return self.callback(*args)


class FakeSDK:
    def __init__(self, frames=(), start_status=0, set_status=0, stop_error=False):
        self.frames = list(frames)
        self.calls = []
        self.kwSetConfig = FakeFunction(self.configure)
        self.kwStartCapture = FakeFunction(lambda index: self.note("start", start_status))
        self.kwStopCapture = FakeFunction(self.stop)
        self.kwResetConfig = FakeFunction(lambda index: self.note("reset", 0))
        self.kwGetForceDataF = FakeFunction(self.read)
        self.set_status, self.stop_error = set_status, stop_error

    def note(self, name, result):
        self.calls.append(name)
        return result

    def configure(self, index, config_ptr):
        config = C.cast(config_ptr, C.POINTER(SensorConfig)).contents
        assert config.decodeMode == 0 and config.linkMode == 0
        assert config.baudRate == 460800
        assert config.paras is not None  # ctypes null pointer object
        assert not bool(config.paras)
        assert config.serialPortName.startswith(b"/dev/")
        return self.note("set", self.set_status)

    def stop(self, index):
        self.calls.append("stop")
        if self.stop_error:
            raise RuntimeError("stop failed")
        return 0

    def read(self, index, values, frame_ptr, byte_ptr):
        self.calls.append("read")
        if not self.frames:
            return 1
        frame, byte_count, raw = self.frames.pop(0)
        for i, value in enumerate(raw):
            values[i] = value
        C.cast(frame_ptr, C.POINTER(C.c_uint64))[0] = frame
        C.cast(byte_ptr, C.POINTER(C.c_size_t))[0] = byte_count
        return 0


def test_sdk_signature_and_64bit_counters():
    frame = 2 ** 40
    sdk = FakeSDK([(frame, frame * 28, [1, 2, 3, 4, 5, 6])])
    reader = KunweiSensor(sdk=sdk, clock=lambda: 20.0)
    assert sdk.kwGetForceDataF.argtypes[-1] == C.POINTER(C.c_size_t)
    assert sdk.kwGetForceDataF.argtypes[-2] == C.POINTER(C.c_uint64)
    assert sdk.kwSetConfig.restype == C.c_int
    reader._poll_once()
    sample = reader.latest()
    assert sample.frame == frame and sample.total_bytes == frame * 28
    assert sample.raw_sensor_6 == pytest.approx(tuple(x * 9.80665 for x in (1, 2, 3, 4, 5, 6)))


def test_duplicate_and_partial_counters_do_not_refresh_timestamp():
    now = [1.0]
    sdk = FakeSDK([(1, 28, [0] * 6), (1, 28, [9] * 6), (1, 56, [8] * 6)])
    reader = KunweiSensor(sdk=sdk, clock=lambda: now[0])
    assert reader._poll_once()
    now[0] = 1.02
    assert not reader._poll_once()
    assert reader.latest().received_monotonic_s == 1.0
    assert not reader._poll_once()
    assert reader.latest().raw_sensor_6 == (0,) * 6
    now[0] = 1.1
    with pytest.raises(StaleSampleError):
        reader.latest(max_age_s=0.05)
    assert reader.snapshot_stats()["duplicate_or_partial"] == 2


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_latches_fault_instead_of_returning_recent_good_sample(bad):
    sdk = FakeSDK([(1, 28, [1] * 6), (2, 56, [bad] + [1] * 5)])
    reader = KunweiSensor(sdk=sdk)
    reader._poll_once()
    with pytest.raises(SensorError, match="nonfinite"):
        reader._poll_once()
    with pytest.raises(SensorError, match="nonfinite"):
        reader.latest()


def test_counter_reset_latches_fault():
    reader = KunweiSensor(sdk=FakeSDK([(9, 252, [0] * 6), (1, 28, [0] * 6)]))
    reader._poll_once()
    with pytest.raises(SensorError, match="regressed"):
        reader._poll_once()
    with pytest.raises(SensorError):
        reader.latest()


def test_finally_closes_capture_on_caller_exception_and_reader_is_not_busy():
    sdk = FakeSDK([(1, 28, [1] * 6)])
    reader = KunweiSensor(sdk=sdk, poll_interval_s=0.01)
    with pytest.raises(ValueError, match="caller"):
        with reader:
            sample = reader.wait_next(timeout_s=0.3)
            assert sample.frame == 1
            time.sleep(0.045)
            assert reader.snapshot_stats()["polls"] < 20
            raise ValueError("caller")
    assert sdk.calls[-2:] == ["stop", "reset"]
    assert not reader._thread.is_alive()


def test_start_failure_stops_and_resets():
    sdk = FakeSDK(start_status=3)
    with pytest.raises(SensorError, match="kwStartCapture"):
        KunweiSensor(sdk=sdk).open()
    assert sdk.calls == ["set", "start", "stop", "reset"]


def test_set_failure_resets_and_stop_failure_does_not_skip_reset():
    sdk = FakeSDK(set_status=4)
    with pytest.raises(SensorError, match="kwSetConfig"):
        KunweiSensor(sdk=sdk).open()
    assert sdk.calls == ["set", "reset"]
    sdk = FakeSDK([(1, 28, [0] * 6)], stop_error=True)
    with pytest.raises(SensorError, match="stop failed"):
        with KunweiSensor(sdk=sdk) as reader:
            reader.wait_next(timeout_s=0.3)
    assert sdk.calls[-2:] == ["stop", "reset"]


def test_no_sample_is_never_a_zero_force_reading():
    reader = KunweiSensor(sdk=FakeSDK())
    with pytest.raises(StaleSampleError, match="no valid"):
        reader.latest()


def test_absolute_library_path_required():
    with pytest.raises(ValueError, match="absolute"):
        KunweiSensor(sdk_path="./libkw-c-lib.so")
