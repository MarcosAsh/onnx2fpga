"""$readmemh images: two's complement at the declared width, element 0 in the low bits."""

import numpy as np


class HexImage:
    def __init__(self, words, bits):
        self.words = [int(w) for w in words]
        self.bits = int(bits)

    @classmethod
    def packed(cls, array, elem_bits):
        """One word per row of the last axis of `array`."""
        array = np.asarray(array, dtype=np.int64)
        lanes = array.shape[-1]
        mask = (1 << elem_bits) - 1
        words = []
        for row in array.reshape(-1, lanes):
            word = 0
            for lane, value in enumerate(row):
                word |= (int(value) & mask) << (lane * elem_bits)
            words.append(word)
        return cls(words, lanes * elem_bits)

    def render(self):
        digits = (self.bits + 3) // 4
        return "".join("%0*x\n" % (digits, word) for word in self.words)

    def write(self, path):
        path.write_text(self.render())
