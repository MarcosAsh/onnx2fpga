"""128-bit DDR records consumed by the fabric replay engine."""

import struct

RECORD = struct.Struct("<QBB6s")


def records(beats):
    """Yield little-endian records; every stream must end at a packet boundary."""
    ended = True
    for data, keep, last in beats:
        if (not isinstance(data, int) or not 0 <= data < 1 << 64
                or not isinstance(keep, int) or not 0 < keep <= 255
                or keep & (keep + 1) or last not in (False, True)
                or (not last and keep != 255)):
            raise ValueError("invalid replay beat")
        ended = bool(last)
        yield RECORD.pack(data, keep, int(last), bytes(6))
    if not ended:
        raise ValueError("replay ends inside a packet")


def read_records(source):
    """Read and validate records from a binary file object."""
    ended = True
    while True:
        record = source.read(RECORD.size)
        if not record:
            break
        if len(record) != RECORD.size:
            raise ValueError("truncated replay record")
        data, keep, last, padding = RECORD.unpack(record)
        if padding != bytes(6) or last not in (0, 1):
            raise ValueError("nonzero replay padding")
        if not 0 < keep <= 255 or keep & (keep + 1) or (not last and keep != 255):
            raise ValueError("invalid replay mask")
        ended = bool(last)
        yield data, keep, ended
    if not ended:
        raise ValueError("replay ends inside a packet")
