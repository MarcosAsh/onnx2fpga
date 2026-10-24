#!/usr/bin/env python3
"""Runs a compiled model on an Icepi Zero over its USB serial port: host.py BUILD_DIR [--frames N]."""

import argparse
import importlib.util
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent


def load_link():
    for candidate in (HERE / "serial_link.py",
                      HERE.parents[2] / "compiler" / "onnx2fpga" / "serial_link.py"):
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location("otf_serial_link", candidate)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise SystemExit("cannot find serial_link.py beside host.py or in the checkout")


serial_link = load_link()
Link, LinkTimeout, Session = serial_link.Link, serial_link.LinkTimeout, serial_link.Session
default_port, read_hex = serial_link.default_port, serial_link.read_hex


def run(link, port, inputs, expected, count, window, timeout):
    failed = 0
    with Session(link, port, timeout) as session:
        started = time.perf_counter()
        try:
            for done, got in enumerate(session.stream(inputs, count, window)):
                want = expected[done % len(expected)]
                if got != want:
                    if failed < 3:
                        print("frame %d differs\n  got  %s\n  want %s"
                              % (done, " ".join("%x" % w for w in got),
                                 " ".join("%x" % w for w in want)))
                    failed += 1
        except LinkTimeout as error:
            print(error)
            return 1
        elapsed = time.perf_counter() - started
        extra = session.leftover()

    rate = count / elapsed
    print("%d frames in %.3f s: %.0f frames/s (the line allows %.0f at %d baud), "
          "%d in flight" % (count, elapsed, rate, link.line_rate, link.baud, window))
    if extra:
        print("%d bytes arrived after the last frame" % len(extra))
    if failed or extra:
        print("FAIL: %d of %d frames differ from golden/expected.hex" % (failed, count))
        return 1
    print("PASS: every frame matches golden/expected.hex bit for bit")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("build", type=pathlib.Path,
                        help="the directory onnx2fpga bitstream wrote")
    parser.add_argument("--port", default=None,
                        help="serial port (default the FT231X under /dev/serial/by-id, "
                             "else /dev/ttyUSB0)")
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--input", type=pathlib.Path, default=None,
                        help="beats to send (default BUILD/golden/input.hex)")
    parser.add_argument("--expected", type=pathlib.Path, default=None,
                        help="beats to expect (default BUILD/golden/expected.hex)")
    parser.add_argument("--window", type=int, default=None,
                        help="frames in flight (default as many as the buffers hold)")
    parser.add_argument("--timeout", type=float, default=2.0,
                        help="seconds without output before giving up")
    args = parser.parse_args(argv)

    link = Link.load(args.build)
    try:
        inputs = Link.frames(read_hex(args.input or args.build / "golden" / "input.hex"),
                             link.in_beats, "the input")
        expected = Link.frames(
            read_hex(args.expected or args.build / "golden" / "expected.hex"),
            link.out_beats, "the expected output")
    except ValueError as error:
        raise SystemExit(str(error))
    if len(inputs) != len(expected):
        raise SystemExit("%d input frames but %d expected" % (len(inputs), len(expected)))
    window = args.window or link.window
    if not 1 <= window <= link.window:
        raise SystemExit("a frame is %d bytes in and %d out but the buffers hold %d and "
                         "%d: rebuild with larger RX_DEPTH or TX_DEPTH"
                         % (link.frame_in, link.frame_out, link.rx_depth, link.tx_depth))
    try:
        import serial  # noqa: F401
    except ImportError:
        raise SystemExit("needs pyserial: pip install pyserial")
    return run(link, args.port or default_port(), inputs, expected, args.frames,
               window, args.timeout)


if __name__ == "__main__":
    sys.exit(main())
