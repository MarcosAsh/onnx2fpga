"""Read ITCH messages from Nasdaq BinaryFILE streams."""

import hashlib

from .itch import HEADER, MTU_PAYLOAD


def messages(stream):
    """Yield length-prefixed payloads; require the session's zero-length terminator."""
    index = 0
    while True:
        header = stream.read(2)
        if len(header) != 2:
            raise ValueError("incomplete BinaryFILE length at message %d" % index)
        length = int.from_bytes(header, "big")
        if length == 0:
            if stream.read(1):
                raise ValueError("data after BinaryFILE session terminator")
            return
        payload = stream.read(length)
        if len(payload) != length:
            raise ValueError("incomplete BinaryFILE payload at message %d" % index)
        yield payload
        index += 1


def select(stream, symbol, limit):
    """Keep a symbol's first messages in source order, starting at its directory entry."""
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("message limit must be a positive integer")
    encoded = symbol.encode("ascii")
    if not 1 <= len(encoded) <= 8 or encoded.strip() != encoded:
        raise ValueError("symbol must contain 1 to 8 ASCII characters without outer spaces")
    locate = None
    selected = []
    digest = hashlib.sha256()
    scanned = 0
    complete = True
    for payload in messages(stream):
        scanned += 1
        digest.update(len(payload).to_bytes(2, "big") + payload)
        if len(payload) < 11:
            raise ValueError("short ITCH header at message %d" % scanned)
        current = int.from_bytes(payload[1:3], "big")
        if payload[:1] == b"R" and len(payload) == 39:
            if payload[11:19].rstrip(b" ") == encoded:
                if locate is not None and locate != current:
                    raise ValueError("symbol changed stock locate within the session")
                locate = current
            elif locate is not None and current == locate:
                raise ValueError("stock locate reassigned within the session")
        if locate is not None and current == locate:
            if len(payload) > MTU_PAYLOAD - HEADER - 2:
                raise ValueError("ITCH message exceeds replay packet capacity")
            selected.append(payload)
            if len(selected) == limit:
                complete = False
                break
    if locate is None:
        raise ValueError("stock directory has no symbol %r" % symbol)
    return selected, {"symbol": symbol, "locate": locate,
                      "source_messages_scanned": scanned,
                      "selected_messages": len(selected), "source_session_complete": complete,
                      "scanned_messages_sha256": digest.hexdigest()}
