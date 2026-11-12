"""Check DMA provisioning without loading a kernel module or mapping memory."""

import fcntl
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from tools.provision_ddr import provision


class ProvisionTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.sysfs, self.devices = self.root / "sys", self.root / "dev"
        self.sysfs.mkdir()
        self.devices.mkdir()
        self.attrs = self.sysfs / "udmabuf0"
        self.page = os.sysconf("SC_PAGE_SIZE")

    def allocation(self):
        self.attrs.mkdir()
        (self.devices / "udmabuf0").touch()
        for name, value in {"dev": "0:0", "phys_addr": "0x100000", "size": str(self.page),
                            "dma_coherent": "0", "sync_for_cpu": "0", "sync_for_device": "0"}.items():
            (self.attrs / name).write_text(value)

    def provision(self, size=16):
        with mock.patch("tools.provision_ddr.stat.S_ISCHR", return_value=True):
            return provision(size, sysfs=self.sysfs, devices=self.devices)

    def test_new_allocation(self):
        def load(*args, **kwargs):
            self.allocation()
        with mock.patch("tools.provision_ddr.subprocess.run", side_effect=load) as run:
            result = self.provision()
        run.assert_called_once_with(["modprobe", "--first-time", "u-dma-buf",
                                     "udmabuf0=%d" % self.page, "dma_mask_bit=31",
                                     "bind=", "udmabuf0_bind="], check=True)
        self.assertEqual(result["bytes"], self.page)
        self.assertEqual(result["address"], 0x100000)
        self.assertEqual((self.attrs / "sync_for_cpu").read_text(), "0")

    def test_existing_allocation_is_untouched(self):
        self.allocation()
        with mock.patch("tools.provision_ddr.subprocess.run") as run:
            self.provision()
            with self.assertRaisesRegex(ValueError, "too small"):
                self.provision(self.page + 1)
        run.assert_not_called()
        self.assertTrue(self.attrs.exists())

    def test_invalid_sizes(self):
        with mock.patch("tools.provision_ddr.subprocess.run") as run:
            for size in (0, -1, 0x80000001, 1.5):
                with self.subTest(size=size), self.assertRaises(ValueError):
                    self.provision(size)
        run.assert_not_called()

    def test_bad_allocation(self):
        self.allocation()
        for name, value, message in (("phys_addr", "0x80000000", "window"),
                                     ("phys_addr", "1", "window"),
                                     ("dma_coherent", "1", "noncoherent"),
                                     ("dev", "1:2", "match sysfs")):
            with self.subTest(name=name, value=value):
                path = self.attrs / name
                original = path.read_text()
                path.write_text(value)
                with self.assertRaisesRegex(ValueError, message):
                    self.provision()
                path.write_text(original)
        self.provision()

    def test_iommu_rejected(self):
        self.allocation()
        (self.attrs / "device" / "iommu_group").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "IOMMU"):
            self.provision()

    def test_busy_allocation(self):
        self.allocation()
        with (self.devices / "udmabuf0").open("rb") as device:
            fcntl.flock(device, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.provision()
        self.provision()

    def test_cache_sync_required(self):
        self.allocation()
        (self.attrs / "sync_for_device").unlink()
        with self.assertRaisesRegex(ValueError, "cache synchronization"):
            self.provision()

    def test_module_failure_is_not_retried(self):
        with mock.patch("tools.provision_ddr.subprocess.run",
                        side_effect=subprocess.CalledProcessError(1, "modprobe")) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                self.provision()
        self.assertEqual(run.call_count, 1)

    def test_creation_timeout_keeps_module_loaded(self):
        with mock.patch("tools.provision_ddr.subprocess.run") as run, \
                mock.patch("tools.provision_ddr.time.monotonic", side_effect=[0, 6]):
            with self.assertRaisesRegex(ValueError, "not unloaded"):
                self.provision()
        self.assertEqual(run.call_count, 1)
