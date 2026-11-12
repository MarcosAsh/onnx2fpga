"""Exclusive userspace access to a persistent Linux u-dma-buf allocation."""

import fcntl
import mmap
import os
from pathlib import Path
import re
import stat

from ..board import BoardError


class DmaBuffer:
    """The caller must stop the DMA engine before acquiring CPU ownership."""

    def __init__(self, name="udmabuf0", *, sysfs="/sys/class/u-dma-buf", devices="/dev"):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("invalid DMA device name")
        self.attributes = Path(sysfs) / name
        self.fd = None
        self.mapping = None
        self.cpu_owned = False
        try:
            self.fd = os.open(Path(devices) / name, os.O_RDWR | os.O_CLOEXEC)
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            info = os.fstat(self.fd)
            major, minor = (int(v) for v in (self.attributes / "dev").read_text().split(":"))
            if not stat.S_ISCHR(info.st_mode) or info.st_rdev != os.makedev(major, minor):
                raise BoardError("DMA device does not match sysfs")
            if int((self.attributes / "dma_coherent").read_text(), 0):
                raise BoardError("HP0 replay requires a noncoherent DMA allocation")
            self.address = int((self.attributes / "phys_addr").read_text(), 0)
            self.size = int((self.attributes / "size").read_text(), 0)
            if self.address < 0 or not 0 < self.size <= 0xFFFFFFF0 or self.size % 16:
                raise BoardError("invalid DMA allocation")
            self.mapping = mmap.mmap(self.fd, self.size, flags=mmap.MAP_SHARED,
                                     prot=mmap.PROT_READ | mmap.PROT_WRITE)
        except Exception:
            if self.mapping is not None:
                self.mapping.close()
            if self.fd is not None:
                os.close(self.fd)
            self.fd = None
            raise

    def _sync(self, owner):
        if self.fd is None:
            raise BoardError("DMA buffer is closed")
        # Inline parameters keep cache synchronization independent of sysfs defaults.
        value = (self.size & 0xFFFFFFF0) | (1 << 2) | 1
        with (self.attributes / ("sync_for_" + owner)).open("w") as target:
            target.write("0x%x\n" % value)

    def acquire(self):
        self._sync("cpu")
        self.cpu_owned = True

    def write(self, payload):
        if not self.cpu_owned or self.fd is None:
            raise BoardError("DMA buffer is not owned by the CPU")
        if len(payload) > self.size:
            raise BoardError("replay exceeds DMA allocation")
        self.mapping[:len(payload)] = payload

    def submit(self):
        if not self.cpu_owned:
            raise BoardError("DMA buffer is not owned by the CPU")
        self.cpu_owned = False
        self._sync("device")

    def close(self):
        if self.fd is None:
            return
        if not self.cpu_owned:
            raise BoardError("stop DMA before closing its buffer")
        self.mapping.close()
        os.close(self.fd)
        self.fd = None
