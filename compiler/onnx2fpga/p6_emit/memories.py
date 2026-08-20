"""Memory images in $readmemh form.

Values are written as two's complement of exactly the declared width, because
that is how the RTL reads them back. Weight words pack SIMD elements with
element 0 in the least significant bits, matching the bit slicing in
otf_mvau.sv.
"""

import numpy as np


class HexImage:
    def __init__(self, words, bits):
        self.words = [int(w) for w in words]
        self.bits = int(bits)

    @property
    def digits(self):
        return (self.bits + 3) // 4

    @classmethod
    def scalar(cls, array, bits):
        mask = (1 << bits) - 1
        flat = np.asarray(array, dtype=np.int64).reshape(-1)
        return cls([int(v) & mask for v in flat], bits)

    @classmethod
    def packed(cls, array, elem_bits):
        """Packs the last axis of `array` into one word per remaining index."""
        array = np.asarray(array, dtype=np.int64)
        lanes = array.shape[-1]
        flat = array.reshape(-1, lanes)
        mask = (1 << elem_bits) - 1
        words = []
        for row in flat:
            word = 0
            for lane, value in enumerate(row):
                word |= (int(value) & mask) << (lane * elem_bits)
            words.append(word)
        return cls(words, lanes * elem_bits)

    def render(self):
        return "".join("%0*x\n" % (self.digits, word) for word in self.words)

    def write(self, path):
        path.write_text(self.render())
        return path

    def __len__(self):
        return len(self.words)
