"""A software model of the tick to trade path, written from the spec, that the RTL must match."""

import struct
from dataclasses import dataclass, field

MIN_LENGTH = 12
FLOW_BITS = 20

ADD, EXECUTE, CANCEL, DELETE, REPLACE = range(5)
HOLD, BUY, SELL = range(3)

# Kind, length and the offsets of shares and price, as in itch_parser.ml.
LAYOUTS = {b"A": (ADD, 36, 20, 32), b"F": (ADD, 40, 20, 32), b"E": (EXECUTE, 31, 19, 0),
           b"C": (EXECUTE, 36, 19, 0), b"X": (CANCEL, 23, 19, 0), b"D": (DELETE, 19, 0, 0),
           b"U": (REPLACE, 35, 27, 31)}


@dataclass
class Config:
    locates: tuple = (2, 3, 5, 6)
    names: tuple = ("BRVO", "CHRL", "ECHO", "FXTR")
    levels: int = 4
    order_bits: int = 12
    spread_shift: int = 6
    flow_shift: int = 5
    buy_threshold: int = 40
    sell_threshold: int = -40
    order_shares: int = 100

    def as_json(self, model):
        return {"locates": list(self.locates), "names": list(self.names),
                "levels": self.levels, "order_bits": self.order_bits,
                "spread_shift": self.spread_shift, "flow_shift": self.flow_shift,
                "buy_threshold": self.buy_threshold, "sell_threshold": self.sell_threshold,
                "order_shares": self.order_shares, "model": model}


@dataclass
class Message:
    data: bytes
    first_beat: int
    last_beat: int


def frame(packet_list):
    """The MoldUDP64 framer, a byte at a time; each message knows the beats of its first and last byte."""
    out, base = [], 0
    for packet in packet_list:
        count, offset = 0, 20
        if len(packet) >= 20:
            count = struct.unpack(">H", packet[18:20])[0]
            count = 0 if count == 0xFFFF else count
        while count > 0 and offset + 2 <= len(packet):
            length = struct.unpack(">H", packet[offset:offset + 2])[0]
            if length < MIN_LENGTH or offset + 2 + length > len(packet):
                break
            start = offset + 2
            out.append(Message(packet[start:start + length], base + start // 8,
                               base + (start + length - 1) // 8))
            offset, count = start + length, count - 1
        base += (len(packet) + 7) // 8
    return out


@dataclass
class Event:
    kind: int
    symbol: int
    side: int
    order: int
    new_order: int
    shares: int
    price: int
    message: Message


def parse(message, locates):
    """The event an ITCH message makes, or None when the parser drops it."""
    data = message.data
    layout = LAYOUTS.get(data[:1])
    locate = int.from_bytes(data[1:3], "big")
    if layout is None or layout[1] != len(data) or locate not in locates:
        return None
    kind, _, shares, price = layout

    def number(offset, size):
        return int.from_bytes(data[offset:offset + size], "big") if offset else 0
    return Event(kind, locates.index(locate), int(data[19:20] == b"S"), number(11, 8),
                 number(19, 8) if kind == REPLACE else 0, number(shares, 4), number(price, 4),
                 message)


def better(ask, a, b):
    return a < b if ask else a > b


@dataclass
class Book:
    """Order table and top of book levels, with the semantics of order_book.ml and book_side.ml."""
    symbols: int
    levels: int
    order_bits: int
    table: dict = field(default_factory=dict)
    collisions: int = 0
    forgotten: int = 0

    def __post_init__(self):
        self.sides = [[[], []] for _ in range(self.symbols)]
        self.flow = [0] * self.symbols

    def slot(self, order):
        return order & ((1 << self.order_bits) - 1)

    @staticmethod
    def remove(levels, price, shares):
        for index, level in enumerate(levels):
            if level[0] == price:
                if shares >= level[1]:
                    del levels[index]
                else:
                    level[1] -= shares
                return

    def insert(self, levels, ask, price, shares, opposite):
        """Adds the order to its level; False when it has no shares or would cross."""
        if shares == 0 or (opposite and (price <= opposite[0][0] if ask
                                         else price >= opposite[0][0])):
            return False
        for level in levels:
            if level[0] == price:
                level[1] = min(level[1] + shares, (1 << 32) - 1)
                return True
        ahead = sum(1 for level in levels if better(ask, level[0], price))
        if ahead < self.levels:
            levels.insert(ahead, [price, shares])
            if len(levels) > self.levels:
                levels.pop()
                self.forgotten += 1
        else:
            self.forgotten += 1
        return True

    def apply(self, event):
        slot_a, slot_b = self.slot(event.order), self.slot(event.new_order)
        entry = self.table.get(slot_a)
        hit = entry is not None and entry[0] == event.order and entry[1] == event.symbol
        sides = self.sides[event.symbol]
        if event.kind == ADD:
            if entry is not None:
                self.collisions += 1
            elif self.insert(sides[event.side], event.side, event.price, event.shares,
                             sides[1 - event.side]):
                self.table[slot_a] = (event.order, event.symbol, event.side, event.price,
                                      event.shares)
        elif hit:
            _, symbol, ask, price, shares = entry
            if event.kind in (EXECUTE, CANCEL):
                executed = min(event.shares, shares)
                self.remove(sides[ask], price, executed)
                if shares > executed:
                    self.table[slot_a] = (event.order, symbol, ask, price, shares - executed)
                else:
                    del self.table[slot_a]
                if event.kind == EXECUTE:
                    amount = min(executed, 0xFFFF) * (1 if ask else -1)
                    flow = self.flow[symbol]
                    self.flow[symbol] = _wrap(flow - (flow >> 3) + amount, FLOW_BITS)
            else:
                self.remove(sides[ask], price, shares)
                del self.table[slot_a]
                if event.kind == REPLACE:
                    if slot_b in self.table:
                        self.collisions += 1
                    elif self.insert(sides[ask], ask, event.price, event.shares,
                                     sides[1 - ask]):
                        self.table[slot_b] = (event.new_order, symbol, ask, event.price,
                                              event.shares)
        return self.snapshot(event.symbol)

    def snapshot(self, symbol):
        def padded(levels):
            return tuple(tuple(level) for level in levels) \
                + ((0, 0),) * (self.levels - len(levels))
        bids, asks = self.sides[symbol]
        return (symbol, padded(bids), padded(asks), self.flow[symbol])


def _wrap(value, bits):
    value &= (1 << bits) - 1
    return value - (1 << bits) if value >> (bits - 1) else value


def imbalance(bid, ask):
    if bid + ask == 0:
        return 0
    n = 127 * (bid - ask)
    q = (2 * abs(n) + bid + ask) // (2 * (bid + ask))
    return -q if n < 0 else q


def _normalized(bid, ask):
    shift = max((bid | ask).bit_length() - 4, 0)
    return imbalance(bid >> shift, ask >> shift)


def features(snapshot, config):
    """The four int8 features of features.ml."""
    _, bids, asks, flow = snapshot
    (bid, bid_shares), (ask, ask_shares) = bids[0], asks[0]
    spread = 127
    if bid_shares and ask_shares:
        spread = min(((ask - bid) & 0xFFFFFFFF) >> config.spread_shift, 127)
    trend = max(-127, min(127, flow >> config.flow_shift))
    depth = _normalized(sum(s for _, s in bids), sum(s for _, s in asks))
    return (spread, _normalized(bid_shares, ask_shares), depth, trend)


def order_frame(token, sell, price, name, shares):
    """The OUCH 5.0 Enter Order of decision.ml."""
    return (b"O" + token.to_bytes(4, "big") + (b"S" if sell else b"B")
            + shares.to_bytes(4, "big") + name.ljust(8).encode() + price.to_bytes(8, "big")
            + b"3YPNN000000" + ("%08X" % token).encode() + b"\0\0")


@dataclass
class Decision:
    snapshot: tuple
    features: tuple
    prediction: int
    action: int
    order: bytes
    message: Message


class Pipeline:
    """Messages to decisions, one per tracked book message; `model` maps four int8s to one."""

    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.book = Book(len(config.locates), config.levels, config.order_bits)
        self.token = 0

    def run(self, messages):
        out = []
        for message in messages:
            event = parse(message, list(self.config.locates))
            if event is not None:
                out.append(self.decide(self.book.apply(event), message))
        return out

    def decide(self, snapshot, message):
        config = self.config
        vector = features(snapshot, config)
        prediction = self.model(vector)
        symbol, bids, asks, _ = snapshot
        action, order = HOLD, b""
        if prediction >= config.buy_threshold and asks[0][1]:
            action = BUY
        elif prediction <= config.sell_threshold and bids[0][1]:
            action = SELL
        if action != HOLD:
            self.token += 1
            price = asks[0][0] if action == BUY else bids[0][0]
            order = order_frame(self.token, action == SELL, price, config.names[symbol],
                                config.order_shares)
        return Decision(snapshot, vector, prediction, action, order, message)
