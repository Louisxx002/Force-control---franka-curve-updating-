"""Driver for the KunWei six-axis F/T sensor on the lab Franka cell.

REAL DRIVER, not a stub. The protocol below was read off the hardware rather
than off a datasheet, because the vendor's own Python samples in
~/Force_sensor/kw-python are configured for a different transport than the one
actually wired up (they expect UDP on 192.168.50.x; this cell has the sensor on
an FT232 USB serial adapter, and the host is not even on that subnet).

WIRE FORMAT, measured on /dev/ttyUSB0 at 460800 8N1
    28-byte frames, streaming continuously at about 1150 Hz:

        offset  0..1    0x48 0xAA        header
                2..25   6 x float32 LE   fx fy fz mx my mz, kgf / kgf*m
                26..27  0x0D 0x0A        terminator

    Current PNP75 wire values are kgf and kgf*m (user confirmed 2026-09-12).
    parse_frame converts all six axes once by 9.80665 to N and Nm.
    Previous comments claiming wire values were N/Nm were incorrect.
    Sending 48AA0D0A starts streaming; 43AA0D0A stops it.

Reading is on a background thread and always returns the most recent complete
frame, so the control loop never blocks on serial I/O. If the loop asks faster
than the sensor produces, it sees the same sample twice, which is correct --
better than waiting.

    python3 kunwei_ft.py --test          read for 3 s and report statistics
    python3 kunwei_ft.py --test --hz 5   also print live values
"""

import struct
import sys
import threading
import time

import numpy as np
from .units import wire_to_si, STANDARD_GRAVITY

DEFAULT_PORT = "/dev/ttyUSB0"
DEFAULT_BAUD = 460800

FRAME_LEN = 28
HEADER = b"\x48\xaa"
TERMINATOR = b"\x0d\x0a"
CMD_START = bytes.fromhex("48aa0d0a")
CMD_STOP = bytes.fromhex("43aa0d0a")


def parse_frame(frame):
    """28 raw bytes -> (force (3,) N, torque (3,) Nm), or None if malformed."""
    if (len(frame) != FRAME_LEN or frame[:2] != HEADER
            or frame[-2:] != TERMINATOR):
        return None
    v = wire_to_si(struct.unpack("<6f", frame[2:26]))
    if not all(np.isfinite(v)):
        return None
    return np.array(v[:3]), np.array(v[3:])


def find_frames(buf):
    """Split a byte buffer into complete frames. Returns (frames, remainder).

    Resynchronises by searching for the header rather than assuming alignment.
    A USB serial link drops and re-buffers bytes; assuming the stream stays
    aligned to a 28-byte boundary works right up until it does not, and then
    produces garbage wrenches that look like real forces.
    """
    frames = []
    i = 0
    while True:
        j = buf.find(HEADER, i)
        if j < 0 or j + FRAME_LEN > len(buf):
            break
        candidate = buf[j:j + FRAME_LEN]
        if candidate[-2:] == TERMINATOR:
            frames.append(candidate)
            i = j + FRAME_LEN
        else:
            i = j + 1          # false header inside the payload, keep looking
    return frames, buf[i:]


class KunweiFT:
    """Threaded reader exposing the latest wrench, in the SENSOR frame.

    The wrench this returns is raw: tool weight and zero offset are still in
    it. Feed it through polish_core.compensate() with a fit from
    ft_gravity_calib.py before using it to close a force loop -- on this cell
    the uncompensated reading contains payload gravity and sensor bias.
    """

    def __init__(self, port=DEFAULT_PORT, baud=DEFAULT_BAUD, timeout=0.05):
        import serial                       # imported here so the module can be
        self._serial = serial               # imported on machines without it
        self.port, self.baud, self.timeout = port, baud, timeout
        self.ser = None
        self._lock = threading.Lock()
        self._force = np.zeros(3)
        self._torque = np.zeros(3)
        self._stamp = 0.0
        self._frames = 0
        self._bad = 0
        self._running = False
        self._thread = None

    # -- lifecycle --
    def open(self):
        self.ser = self._serial.Serial(self.port, self.baud, timeout=self.timeout)
        self.ser.reset_input_buffer()
        try:
            self.ser.write(CMD_START)       # best effort; it usually streams already
        except Exception:
            pass
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        if not self.wait_ready():
            self.close()
            raise RuntimeError(
                f"no valid frames from {self.port} within 2 s. Check the sensor "
                f"is powered, the FT232 adapter is plugged in, and that nothing "
                f"else has the port open.")
        return self

    def close(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(CMD_STOP)
            except Exception:
                pass
            self.ser.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    # -- reading --
    def _reader(self):
        buf = b""
        while self._running:
            try:
                # Read ONLY what has already arrived. A plain read(4096) blocks
                # until the buffer fills or the timeout expires, so it hands
                # back data in timeout-sized batches -- measured on this cell,
                # that made every sample 25 ms stale on average and 50 ms at
                # p99, on a stream that is actually 1 kHz. The wrench looked
                # perfect and was a quarter of a control period out of date.
                # read(1) still blocks, but only until the next frame, ~1 ms.
                waiting = self.ser.in_waiting
                chunk = self.ser.read(waiting if waiting else 1)
            except Exception:
                break
            if not chunk:
                continue
            buf += chunk
            frames, buf = find_frames(buf)
            if len(buf) > 8 * FRAME_LEN:     # nothing parseable, do not grow
                buf = buf[-FRAME_LEN:]
            for fr in frames:
                parsed = parse_frame(fr)
                if parsed is None:
                    self._bad += 1
                    continue
                with self._lock:
                    self._force, self._torque = parsed
                    self._stamp = time.monotonic()
                    self._frames += 1

    def wait_ready(self, timeout=2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._frames > 0:
                return True
            time.sleep(0.01)
        return False

    def read(self):
        """-> (force (3,) N, torque (3,) Nm), most recent complete frame."""
        with self._lock:
            return self._force.copy(), self._torque.copy()

    def read_force(self):
        with self._lock:
            return self._force.copy()

    @property
    def age(self):
        """Seconds since the last good frame. Watch this in a control loop:
        a sensor that has silently stopped still returns its last value."""
        with self._lock:
            return time.monotonic() - self._stamp if self._stamp else float("inf")

    @property
    def stats(self):
        return {"frames": self._frames, "bad": self._bad}


def _self_check():
    """Frame parsing, offline. Runs anywhere, no sensor needed."""
    f = np.array([0.375, -1.669, 1.213], dtype=np.float32)
    m = np.array([-0.0597, -0.0122, -0.0117], dtype=np.float32)
    good = HEADER + struct.pack("<6f", *f, *m) + TERMINATOR
    assert len(good) == FRAME_LEN

    pf, pm = parse_frame(good)
    assert np.allclose(pf, f * STANDARD_GRAVITY, atol=1e-6) and np.allclose(pm, m * STANDARD_GRAVITY, atol=1e-6)

    assert parse_frame(good[:-1]) is None, "short frame must be rejected"
    assert parse_frame(b"\x00\x00" + good[2:]) is None, "bad header rejected"
    assert parse_frame(good[:-2] + b"\x00\x00") is None, "bad terminator rejected"
    nan = HEADER + struct.pack("<6f", *([float("nan")] * 6)) + TERMINATOR
    assert parse_frame(nan) is None, "NaN payload must be rejected"

    # a stream that starts mid-frame must resynchronise, not emit garbage
    stream = b"\x9a\x3f\xbe\x22" + good * 5
    frames, rest = find_frames(stream)
    assert len(frames) == 5, f"expected 5 frames, got {len(frames)}"
    assert all(parse_frame(x) is not None for x in frames)

    # a partial frame at the end must be kept as remainder, not dropped
    frames, rest = find_frames(good * 2 + good[:10])
    assert len(frames) == 2 and rest == good[:10]

    # a 0x48 0xAA appearing INSIDE a payload must not derail the parser
    tricky = HEADER + struct.pack("<6f", 0, 0, 0, 0, 0, 0) + TERMINATOR
    payload_with_header = HEADER + b"\x48\xaa" + b"\x00" * 22 + TERMINATOR
    frames, _ = find_frames(payload_with_header + tricky)
    assert all(len(x) == FRAME_LEN for x in frames)
    print("  ok  frame parse, rejection, resync and remainder handling")
    print("kunwei_ft: offline self-check passed")


def _test(duration=3.0, print_hz=0.0):
    with KunweiFT() as ft:
        t0 = time.monotonic()
        samples = []
        next_print = 0.0
        while time.monotonic() - t0 < duration:
            f, m = ft.read()
            samples.append(np.concatenate([f, m]))
            now = time.monotonic() - t0
            if print_hz and now >= next_print:
                next_print = now + 1.0 / print_hz
                print(f"  t={now:5.2f}s  F=[{f[0]:+7.3f} {f[1]:+7.3f} {f[2]:+7.3f}] N"
                      f"  M=[{m[0]:+7.4f} {m[1]:+7.4f} {m[2]:+7.4f}] Nm")
            time.sleep(0.001)
        a = np.array(samples)
        st = ft.stats
        print(f"\n  frames {st['frames']} in {duration:.1f}s "
              f"= {st['frames'] / duration:.0f} Hz, malformed {st['bad']}")
        print(f"  age of last frame: {ft.age * 1000:.1f} ms")
        names = ["fx", "fy", "fz", "mx", "my", "mz"]
        print("  axis      mean       std       min       max")
        for i, n in enumerate(names):
            print(f"  {n:>4}  {a[:, i].mean():+9.4f} {a[:, i].std():9.5f} "
                  f"{a[:, i].min():+9.4f} {a[:, i].max():+9.4f}")
        print(f"\n  |F| standing = {np.linalg.norm(a[:, :3].mean(axis=0)):.3f} N "
              f"-- this is tool weight plus zero offset, NOT contact force. "
              f"Calibrate it out with ft_gravity_calib.py before closing a loop.")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    elif "--test" in sys.argv:
        hz = 0.0
        if "--hz" in sys.argv:
            hz = float(sys.argv[sys.argv.index("--hz") + 1])
        _test(print_hz=hz)
    else:
        print(__doc__.strip().split("\n\n")[0])
        print("\n  --self-check   parse/resync tests, no hardware needed"
              "\n  --test         read the real sensor for 3 s and report"
              "\n  --test --hz 5  also print live values at 5 Hz")
