"""Synthetic NASDAQ TotalView ITCH 5.0 traffic in MoldUDP64 packets, seeded and deterministic."""

import random
import struct

TICK = 100          # one cent, in ITCH's four implied decimals
SESSION = b"ONNX2FPGA1"
HEADER = 20
MTU_PAYLOAD = 1400

# Stock locate, symbol and opening price for each stock in the synthetic market.
SYMBOLS = ((1, "ALFA", 250000), (2, "BRVO", 1015000), (3, "CHRL", 432100),
           (4, "DLTA", 87500), (5, "ECHO", 1999900), (6, "FXTR", 55000))

LENGTHS = {"S": 12, "R": 39, "A": 36, "F": 40, "E": 31, "C": 36, "X": 23, "D": 19,
           "U": 35, "P": 44}


def _header(kind, locate, timestamp):
    return struct.pack(">cHH", kind.encode(), locate, 0) + timestamp.to_bytes(6, "big")


def _stock(name):
    return name.ljust(8).encode()


def system_event(timestamp, code):
    return _header("S", 0, timestamp) + code.encode()


def stock_directory(locate, timestamp, name):
    return (_header("R", locate, timestamp) + _stock(name)
            + struct.pack(">ccIcc2scccccIc", b"Q", b"N", 100, b"N", b"C", b"Z ", b"P",
                          b"N", b"N", b"2", b"N", 0, b"N"))


def add_order(locate, timestamp, order, side, shares, name, price, attribution=None):
    kind = "A" if attribution is None else "F"
    body = _header(kind, locate, timestamp) + struct.pack(
        ">QcI8sI", order, side.encode(), shares, _stock(name), price)
    return body if attribution is None else body + attribution.encode()


def order_executed(locate, timestamp, order, shares, match, price=None):
    if price is None:
        return _header("E", locate, timestamp) + struct.pack(">QIQ", order, shares, match)
    return _header("C", locate, timestamp) + struct.pack(">QIQcI", order, shares, match,
                                                          b"Y", price)


def order_cancel(locate, timestamp, order, shares):
    return _header("X", locate, timestamp) + struct.pack(">QI", order, shares)


def order_delete(locate, timestamp, order):
    return _header("D", locate, timestamp) + struct.pack(">Q", order)


def order_replace(locate, timestamp, order, new_order, shares, price):
    return _header("U", locate, timestamp) + struct.pack(">QQII", order, new_order, shares,
                                                          price)


def trade(locate, timestamp, side, shares, name, price, match):
    return _header("P", locate, timestamp) + struct.pack(
        ">QcI8sIQ", 0, side.encode(), shares, _stock(name), price, match)


class _Book:
    """One stock's true book in the generator, with price-time priority."""

    def __init__(self, locate, name, price):
        self.locate, self.name, self.reference = locate, name, price
        self.orders = {}                  # order -> [side, price, shares]
        self.levels = ({}, {})            # per side, price -> orders in arrival order
        self.live = []
        self.where = {}

    def best(self, side):
        prices = self.levels[side]
        if not prices:
            return None
        return max(prices) if side == 0 else min(prices)

    def add(self, order, side, price, shares):
        self.orders[order] = [side, price, shares]
        self.levels[side].setdefault(price, []).append(order)
        self.where[order] = len(self.live)
        self.live.append(order)

    def remove(self, order):
        side, price, _ = self.orders.pop(order)
        queue = self.levels[side][price]
        queue.remove(order)
        if not queue:
            del self.levels[side][price]
        index = self.where.pop(order)
        last = self.live.pop()
        if last != order:
            self.live[index] = last
            self.where[last] = index


class Generator:
    """Order flow for a few stocks, in roughly the mix of a NASDAQ day, plus untracked noise."""

    MIX = (("add", 42), ("delete", 30), ("replace", 12), ("execute", 9), ("cancel", 5),
           ("trade", 2))
    LOTS = (100, 100, 100, 100, 100, 200, 200, 300, 400, 500, 1000, 2500)

    def __init__(self, seed=1, symbols=SYMBOLS):
        self.rng = random.Random(seed)
        self.books = [_Book(*symbol) for symbol in symbols]
        self.timestamp = 34200 * 10 ** 9
        self.order = 1000
        self.match = 1
        self.kinds = [kind for kind, weight in self.MIX for _ in range(weight)]

    def _tick(self):
        self.timestamp += self.rng.randint(20, 4000)
        return self.timestamp

    def _price(self, book, side):
        """A resting price that never crosses the other side, mostly a few ticks from the touch."""
        rng = self.rng
        own, other = book.best(side), book.best(1 - side)
        sign = 1 if side == 0 else -1
        if own is None:
            anchor = other + sign * TICK if other is not None else book.reference
        else:
            anchor = own
        distance = 0
        while rng.random() < 0.55 and distance < 30:
            distance += 1
        price = anchor - sign * distance * TICK
        if own is not None and rng.random() < 0.15:
            price = own + sign * TICK
        if other is not None:
            price = min(price, other - TICK) if side == 0 else max(price, other + TICK)
        return max(price, TICK)

    def _shares(self):
        return self.rng.choice(self.LOTS) if self.rng.random() < 0.9 \
            else self.rng.randint(1, 99)

    def _new_order(self):
        self.order += self.rng.choice((1, 1, 1, 2, 3))
        return self.order

    def _kind(self, book):
        live = len(book.live)
        if live < 10:
            return "add"
        if live > 150:
            return self.rng.choice(("delete", "delete", "execute", "replace"))
        return self.rng.choice(self.kinds)

    def step(self):
        """One message, as bytes."""
        rng = self.rng
        book = rng.choice(self.books)
        kind = self._kind(book)
        timestamp = self._tick()
        if kind == "add":
            side = rng.randint(0, 1)
            price, shares, order = self._price(book, side), self._shares(), self._new_order()
            book.add(order, side, price, shares)
            attribution = "NSDQ" if rng.random() < 0.06 else None
            return add_order(book.locate, timestamp, order, "BS"[side], shares, book.name,
                             price, attribution)
        if kind == "trade":
            self.match += 1
            return trade(book.locate, timestamp, rng.choice("BS"), self._shares(), book.name,
                         book.reference, self.match)
        if kind == "execute":
            sides = [side for side in (0, 1) if book.levels[side]]
            side = rng.choice(sides)
            order = book.levels[side][book.best(side)][0]
            _, price, shares = book.orders[order]
            executed = shares if rng.random() < 0.6 else rng.randint(1, shares)
            self.match += 1
            if executed == shares:
                book.remove(order)
            else:
                book.orders[order][2] -= executed
            book.reference = price
            return order_executed(book.locate, timestamp, order, executed, self.match,
                                  price if rng.random() < 0.15 else None)
        order = rng.choice(book.live)
        side, price, shares = book.orders[order]
        if kind == "cancel" and shares > 1:
            cancelled = rng.randint(1, shares - 1)
            book.orders[order][2] -= cancelled
            return order_cancel(book.locate, timestamp, order, cancelled)
        if kind == "replace":
            book.remove(order)
            new_order, new_price, new_shares = self._new_order(), self._price(book, side), \
                self._shares()
            book.add(new_order, side, new_price, new_shares)
            return order_replace(book.locate, timestamp, order, new_order, new_shares,
                                 new_price)
        book.remove(order)
        return order_delete(book.locate, timestamp, order)

    def messages(self, count):
        """`count` messages: the start of day, a directory of the stocks, then order flow."""
        out = [system_event(self._tick(), "O")]
        out += [stock_directory(book.locate, self._tick(), book.name) for book in self.books]
        out.append(system_event(self._tick(), "Q"))
        while len(out) < count:
            out.append(self.step())
        return out


def packets(messages, seed=1):
    """MoldUDP64 packets up to an Ethernet MTU, with heartbeats and an end of session packet."""
    rng = random.Random(seed)
    out, sequence, index = [], 1, 0
    while index < len(messages):
        if rng.random() < 0.01:
            out.append(SESSION + struct.pack(">QH", sequence, 0))
        wanted, block = rng.randint(1, 40), b""
        count = 0
        while index < len(messages) and count < wanted \
                and HEADER + len(block) + 2 + len(messages[index]) <= MTU_PAYLOAD:
            block += struct.pack(">H", len(messages[index])) + messages[index]
            index += 1
            count += 1
        out.append(SESSION + struct.pack(">QH", sequence, count) + block)
        sequence += count
    out.append(SESSION + struct.pack(">QH", sequence, 0xFFFF))
    return out


def beats(packet_list):
    """The 64 bit stream: (data, keep, last) per beat, byte 0 of a beat in the low bits."""
    out = []
    for packet in packet_list:
        for start in range(0, len(packet), 8):
            chunk = packet[start:start + 8]
            out.append((int.from_bytes(chunk, "little"), (1 << len(chunk)) - 1,
                        start + 8 >= len(packet)))
    return out


def write_beats(path, beat_list):
    """Ten bytes a beat: the data little endian, then keep, then last."""
    with open(path, "wb") as file:
        for data, keep, last in beat_list:
            file.write(data.to_bytes(8, "little") + bytes((keep, int(last))))
