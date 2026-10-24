"""An ONNX to streaming-dataflow FPGA compiler; the public API is in onnx2fpga.api."""

from . import errors
from .api import Accelerator, from_keras, load
from .errors import Onnx2FpgaError
from .hardware_layer import HardwareLayer

__version__ = "0.1.0"

__all__ = ["Accelerator", "HardwareLayer", "Onnx2FpgaError", "errors", "from_keras",
           "load"]
