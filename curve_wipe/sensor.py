"""Read-only Kunwei SDK adapter; never zeros the sensor or operates a robot.

SDK counters establish newly received data, not the age inside a device buffer.
All timestamps here use the host monotonic clock and denote receipt by this reader.
"""
from __future__ import annotations

import argparse
import ctypes as C
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import threading
import time
from .units import wire_to_si

DEFAULT_PORT = "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_BG033HFP-if00-port0"
DEFAULT_SDK = "/home/pnp/Downloads/kw_sdk_general_v1.0.4/lib/linux_amd64/libkw-c-lib.so"


class SensorError(RuntimeError):
    pass


class StaleSampleError(SensorError):
    pass


class SensorConfig(C.Structure):
    # Exact ABI from include/kwcapture.h, including pointer and ushort fields.
    _fields_ = [("linkMode", C.c_int), ("decodeMode", C.c_int),
                ("paras", C.POINTER(C.c_float)), ("sensorIp", C.c_char_p),
                ("sensorPort", C.c_ushort), ("localIp", C.c_char_p),
                ("localPort", C.c_ushort), ("serialPortName", C.c_char_p),
                ("baudRate", C.c_int)]


@dataclass(frozen=True)
class ForceSample:
    raw_sensor_6: tuple[float, ...]
    frame: int
    total_bytes: int
    received_monotonic_s: float

    def age_s(self, now=None):
        return (time.monotonic() if now is None else now) - self.received_monotonic_s


def _bind_sdk(sdk):
    signatures = {
        "kwSetConfig": ([C.c_int, C.POINTER(SensorConfig)], C.c_int),
        "kwStartCapture": ([C.c_int], C.c_int),
        "kwStopCapture": ([C.c_int], C.c_int),
        "kwResetConfig": ([C.c_int], C.c_int),
        "kwGetForceDataF": ([C.c_int, C.POINTER(C.c_float),
                             C.POINTER(C.c_uint64), C.POINTER(C.c_size_t)], C.c_int),
    }
    for name, (args, result) in signatures.items():
        function = getattr(sdk, name)
        function.argtypes, function.restype = args, result
    return sdk


class KunweiSensor:
    """Background latest-sample reader. Construction/import does not open hardware.

    Both frame and byte counters must increase before a sample refreshes the
    timestamp. Counter regression and nonfinite data latch a reader fault.
    A nonzero SDK read status does not refresh data; latest() then ages out.
    """
    def __init__(self, port=DEFAULT_PORT, baud=460800, sdk_path=DEFAULT_SDK,
                 device_index=0, poll_interval_s=0.002, *, sdk=None,
                 clock=time.monotonic):
        if not Path(sdk_path).is_absolute():
            raise ValueError("sdk_path must be absolute")
        if not math.isfinite(poll_interval_s) or poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")
        self.port, self.baud = str(port), int(baud)
        self.sdk_path, self.device_index = str(sdk_path), int(device_index)
        self.poll_interval_s, self._clock = poll_interval_s, clock
        self._sdk = _bind_sdk(sdk) if sdk is not None else None
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread = None
        self._sample = None
        self._fault = None
        self._configured = False
        self._start_attempted = False
        self._used = False
        self._last_counters = (0, 0)
        self._stats = dict(polls=0, accepted=0, duplicate_or_partial=0,
                           sdk_read_errors=0, nonfinite=0, counter_regressions=0)
        self._cleanup_errors = []
        self._port_bytes = self.port.encode()
        self._config = SensorConfig(linkMode=0, decodeMode=0, paras=None,
                                    sensorIp=None, sensorPort=5152,
                                    localIp=None, localPort=8886,
                                    serialPortName=self._port_bytes, baudRate=self.baud)

    def open(self):
        if self._used:
            raise SensorError("create a new KunweiSensor for each capture")
        self._used = True
        if self._sdk is None:
            self._sdk = _bind_sdk(C.CDLL(self.sdk_path))
        try:
            # Reset even if SetConfig partially fails inside the vendor library.
            self._configured = True
            status = self._sdk.kwSetConfig(self.device_index, C.byref(self._config))
            if status != 0:
                raise SensorError(f"kwSetConfig failed: {status}")
            self._start_attempted = True
            status = self._sdk.kwStartCapture(self.device_index)
            if status != 0:
                raise SensorError(f"kwStartCapture failed: {status}")
            self._thread = threading.Thread(target=self._reader, daemon=True,
                                            name="kunwei-raw-reader")
            self._thread.start()
        except BaseException:
            self._cleanup()
            raise
        return self

    def _cleanup(self):
        try:
            if self._start_attempted:
                self._start_attempted = False
                result = self._sdk.kwStopCapture(self.device_index)
                if result != 0:
                    self._cleanup_errors.append(f"kwStopCapture: {result}")
        except Exception as exc:
            self._cleanup_errors.append(f"kwStopCapture: {exc}")
        finally:
            if self._configured:
                self._configured = False
                try:
                    result = self._sdk.kwResetConfig(self.device_index)
                    if result != 0:
                        self._cleanup_errors.append(f"kwResetConfig: {result}")
                except Exception as exc:
                    self._cleanup_errors.append(f"kwResetConfig: {exc}")

    def _poll_once(self):
        values = (C.c_float * 6)()
        frame, byte_count = C.c_uint64(), C.c_size_t()
        status = self._sdk.kwGetForceDataF(self.device_index, values,
                                         C.byref(frame), C.byref(byte_count))
        now = self._clock()
        with self._condition:
            self._stats["polls"] += 1
            if status != 0:
                self._stats["sdk_read_errors"] += 1
                return False
            counters = (frame.value, byte_count.value)
            last_frame, last_bytes = self._last_counters
            if counters[0] < last_frame or counters[1] < last_bytes:
                self._stats["counter_regressions"] += 1
                self._fault = "SDK counters regressed; capture must be restarted"
                self._condition.notify_all()
                raise SensorError(self._fault)
            if counters[0] <= last_frame or counters[1] <= last_bytes:
                self._stats["duplicate_or_partial"] += 1
                return False
            # Record counters before validating the payload: the same invalid
            # frame must never reappear as an apparently new valid reading.
            self._last_counters = counters
            raw = wire_to_si(values)
            if not all(math.isfinite(x) for x in raw) or not math.isfinite(now):
                self._stats["nonfinite"] += 1
                self._fault = "nonfinite force data or clock"
                self._condition.notify_all()
                raise SensorError(self._fault)
            self._sample = ForceSample(raw, *counters, now)
            self._stats["accepted"] += 1
            self._condition.notify_all()
            return True

    def _reader(self):
        try:
            while not self._stop.is_set():
                self._poll_once()
                # Event.wait is interruptible and prevents busy polling.
                self._stop.wait(self.poll_interval_s)
        except Exception as exc:
            with self._condition:
                self._fault = str(exc)
                self._condition.notify_all()
        finally:
            self._cleanup()

    def latest(self, max_age_s=0.05):
        if not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError("max_age_s must be positive")
        with self._condition:
            if self._fault:
                raise SensorError(self._fault)
            if self._sample is None:
                raise StaleSampleError("no valid sensor sample received")
            age = self._sample.age_s(self._clock())
            if not math.isfinite(age) or age < 0 or age > max_age_s:
                raise StaleSampleError(f"sensor sample age {age:.6f} s exceeds {max_age_s} s")
            return self._sample

    def wait_next(self, after_frame=0, timeout_s=0.1, max_age_s=0.05):
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        with self._condition:
            self._condition.wait_for(
                lambda: self._fault or (self._sample is not None
                                       and self._sample.frame > after_frame),
                timeout=timeout_s)
            sample = self.latest(max_age_s)
            if sample.frame <= after_frame:
                raise StaleSampleError("no new sensor frame before timeout")
            return sample

    def snapshot_stats(self):
        with self._condition:
            return {**self._stats,
                    "last_sample_age_s": None if self._sample is None else
                    self._sample.age_s(self._clock()),
                    "fault": self._fault,
                    "cleanup_errors": list(self._cleanup_errors)}

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                # Do not reset configuration while a foreign function uses it.
                raise SensorError("SDK read is blocked; reader will clean up when it returns")
        else:
            self._cleanup()
        if self._cleanup_errors:
            raise SensorError("; ".join(self._cleanup_errors))

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except SensorError:
            if exc is None:
                raise
        return False


def record_raw(seconds=3.0, *, port=DEFAULT_PORT, baud=460800,
               sdk_path=DEFAULT_SDK, max_age_s=0.05):
    """Explicit hardware operation: capture raw data, without tare/calibration."""
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be finite and positive")
    reader = KunweiSensor(port=port, baud=baud, sdk_path=sdk_path)
    samples, ages, stale_waits, fault = [], [], 0, None
    started = time.monotonic()
    try:
        with reader:
            last_frame = 0
            while time.monotonic() - started < seconds:
                try:
                    sample = reader.wait_next(last_frame, timeout_s=0.05,
                                              max_age_s=max_age_s)
                except StaleSampleError:
                    stale_waits += 1
                    continue
                last_frame = sample.frame
                age = sample.age_s()
                ages.append(age)
                samples.append({**asdict(sample), "observed_age_s": age})
    except SensorError as exc:
        fault = str(exc)
    return {"mode": "raw_sensor_recording_only", "port": port, "baud": baud,
            "frame_bytes": 28, "sdk_path": sdk_path,
            "wrench_order": ["Fx", "Fy", "Fz", "Mx", "My", "Mz"],
            "units": ["N", "N", "N", "Nm", "Nm", "Nm"],
            "timestamp_basis": "host monotonic receipt, not device acquisition",
            "duration_s": time.monotonic() - started,
            "samples": samples,
            "statistics": {**reader.snapshot_stats(), "recorded": len(samples),
                           "stale_waits": stale_waits,
                           "max_observed_age_s": max(ages) if ages else None,
                           "mean_observed_age_s": sum(ages) / len(ages) if ages else None},
            "error": fault}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baud", type=int, default=460800)
    parser.add_argument("--sdk-path", default=DEFAULT_SDK)
    parser.add_argument("--max-age-s", type=float, default=0.05)
    args = parser.parse_args(argv)
    report = record_raw(args.seconds, port=args.port, baud=args.baud,
                        sdk_path=args.sdk_path, max_age_s=args.max_age_s)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "statistics": report["statistics"],
                      "error": report["error"]}, indent=2, allow_nan=False))
    return 1 if report["error"] or not report["samples"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
