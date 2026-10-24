"""The byte protocol between a host and a compiled model behind a UART; standard library only."""

import glob
import json
import pathlib
import time


def read_hex(path):
    return [int(line, 16) for line in pathlib.Path(path).read_text().split()]


class LinkTimeout(Exception):
    """The board stopped answering. Carries how far the run got."""

    def __init__(self, timeout, done, count, partial, baud):
        self.done, self.count, self.partial = done, count, partial
        Exception.__init__(
            self, "no output for %.1f s after %d of %d frames (%d bytes of a partial "
                  "frame): is the bitstream loaded and the baud rate %d?"
                  % (timeout, done, count, partial, baud))


class Link:
    """The byte framing, from the manifest and board.json."""

    def __init__(self, manifest, board):
        def port(side):
            described = manifest[side]
            width = described["elem_bits"] * described["elems_per_beat"]
            return (width + 7) // 8, described["beats"]

        self.in_bytes, self.in_beats = port("input")
        self.out_bytes, self.out_beats = port("output")
        self.baud = board["baud"]
        self.rx_depth = board["rx_depth"]
        self.tx_depth = board["tx_depth"]

    @classmethod
    def load(cls, build, board="icepi_zero"):
        build = pathlib.Path(build)
        manifest = json.loads((build / "manifest.json").read_text())
        settings = json.loads((build / board / "board.json").read_text())
        return cls(manifest, settings)

    @property
    def frame_in(self):
        return self.in_bytes * self.in_beats

    @property
    def frame_out(self):
        return self.out_bytes * self.out_beats

    @property
    def window(self):
        """Frames that may be in flight at once, so neither buffer can overflow."""
        return min(self.rx_depth // self.frame_in, self.tx_depth // self.frame_out)

    @property
    def line_rate(self):
        """Frames per second the serial line allows, ten bits to a byte."""
        return self.baud / 10 / max(self.frame_in, self.frame_out)

    def encode(self, beats):
        return b"".join(beat.to_bytes(self.in_bytes, "little") for beat in beats)

    def decode(self, data):
        size = self.out_bytes
        return [int.from_bytes(data[k:k + size], "little")
                for k in range(0, len(data) - len(data) % size, size)]

    @staticmethod
    def frames(beats, per_frame, what):
        if not beats or len(beats) % per_frame:
            raise ValueError("%s holds %d beats, not a whole number of %d beat frames"
                             % (what, len(beats), per_frame))
        return [beats[k:k + per_frame] for k in range(0, len(beats), per_frame)]


def default_port():
    found = sorted(glob.glob("/dev/serial/by-id/*FT231X*"))
    return found[0] if found else "/dev/ttyUSB0"


class Session:
    """An open serial line to a board; `opener` makes the line from (port, baud), pyserial Serial by default."""

    def __init__(self, link, port=None, timeout=2.0, opener=None):
        self.link = link
        self.port = port or default_port()
        self.timeout = timeout
        self.opener = opener
        self.line = None
        self._pending = bytearray()

    # Programming re-enumerates the FT231X, so wait for udev to recreate the port.
    SETTLE_SECONDS = 5.0

    def open(self):
        opener = self.opener
        if opener is None:
            import serial

            def opener(port, baud):
                return serial.Serial(port, baud, timeout=0.05)
        deadline = time.monotonic() + self.SETTLE_SECONDS
        while True:
            try:
                self.line = opener(self.port, self.link.baud)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.1)
        self.reset()
        return self

    def reset(self):
        """A break clears whatever a previous run left half sent."""
        self.line.break_condition = True
        time.sleep(0.01)
        self.line.break_condition = False
        time.sleep(0.01)
        self.line.reset_input_buffer()

    def close(self):
        if self.line is not None:
            self.line.close()
            self.line = None

    def __enter__(self):
        return self.open() if self.line is None else self

    def __exit__(self, *exc):
        self.close()

    def stream(self, inputs, count=None, window=None):
        """Sends the frames of `inputs` in turn and yields each output frame, with at most `window` in flight."""
        link, line = self.link, self.line
        count = len(inputs) if count is None else count
        window = window or link.window
        payloads = [link.encode(frame) for frame in inputs]
        sent = done = 0
        pending = self._pending = bytearray()
        last_progress = time.perf_counter()
        while done < count:
            while sent < count and sent - done < window:
                line.write(payloads[sent % len(payloads)])
                sent += 1
            pending += line.read(max(1, line.in_waiting))
            while len(pending) >= link.frame_out and done < count:
                yield link.decode(bytes(pending[:link.frame_out]))
                del pending[:link.frame_out]
                done += 1
                last_progress = time.perf_counter()
            if time.perf_counter() - last_progress > self.timeout:
                raise LinkTimeout(self.timeout, done, count, len(pending), link.baud)

    def leftover(self, settle=0.05):
        """Bytes after the last expected frame, which a correct board never sends."""
        time.sleep(settle)
        line = self.line
        late = line.read(max(1, line.in_waiting)) if line.in_waiting else b""
        return bytes(self._pending) + late

    def run(self, inputs, count=None, window=None):
        """Every output frame for `inputs`, sent once each."""
        return list(self.stream(inputs, count, window))
