#!/usr/bin/env python3
"""Provision a persistent u-dma-buf allocation for KR260 DDR replay."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import stat
import subprocess
import time


def inspect(size, sysfs, devices):
    attrs = Path(sysfs) / "udmabuf0"
    fd = os.open(Path(devices) / "udmabuf0", os.O_RDWR | os.O_CLOEXEC)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        device = os.fstat(fd)
        major, minor = (int(value) for value in (attrs / "dev").read_text().split(":"))
        if not stat.S_ISCHR(device.st_mode) or device.st_rdev != os.makedev(major, minor):
            raise ValueError("DMA device does not match sysfs")
        address = int((attrs / "phys_addr").read_text(), 0)
        available = int((attrs / "size").read_text(), 0)
        if available < size:
            raise ValueError("existing DMA buffer is too small; stop the fabric before replacing it")
        if address < 0 or address % 256 or available % 16 or address + available > 0x80000000:
            raise ValueError("DMA allocation is outside the aligned low 2 GiB window")
        if int((attrs / "dma_coherent").read_text(), 0):
            raise ValueError("HP0 replay requires a noncoherent DMA allocation")
        if (attrs / "device" / "iommu_group").exists():
            raise ValueError("IOMMU-backed DMA allocation is unsupported")
        for name in ("sync_for_cpu", "sync_for_device"):
            if not (attrs / name).is_file():
                raise ValueError("missing DMA cache synchronization attribute: %s" % name)
        return {"device": str(Path(devices) / "udmabuf0"), "address": address,
                "bytes": available, "physical_validation": "not_run"}
    finally:
        os.close(fd)


def provision(size, *, sysfs="/sys/class/u-dma-buf", devices="/dev", modprobe="modprobe"):
    if not isinstance(size, int) or not 0 < size <= 0x80000000:
        raise ValueError("DMA size must be between 1 byte and 2 GiB")
    page = os.sysconf("SC_PAGE_SIZE")
    size = ((size + page - 1) // page) * page
    attrs = Path(sysfs) / "udmabuf0"
    device = Path(devices) / "udmabuf0"
    if not attrs.exists() and not device.exists():
        subprocess.run([modprobe, "--first-time", "u-dma-buf", "udmabuf0=%d" % size,
                        "dma_mask_bit=31", "bind=", "udmabuf0_bind="], check=True)
        deadline = time.monotonic() + 5
        while not attrs.exists() or not device.exists():
            if time.monotonic() >= deadline:
                raise ValueError("u-dma-buf did not create udmabuf0; allocation was not unloaded")
            time.sleep(.05)
    return inspect(size, sysfs, devices)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    size = parser.add_mutually_exclusive_group(required=True)
    size.add_argument("--bytes", type=lambda value: int(value, 0))
    size.add_argument("--records", type=Path, help="allocate enough space for this packed replay")
    args = parser.parse_args()
    requested = args.records.stat().st_size if args.records is not None else args.bytes
    print(json.dumps(provision(requested), indent=2))


if __name__ == "__main__":
    main()
