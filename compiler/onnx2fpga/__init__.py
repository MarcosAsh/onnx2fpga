"""An ONNX to streaming-dataflow FPGA compiler; the public API is in onnx2fpga.api."""

__version__ = "0.1.0"

from . import errors  # noqa: E402
from .api import Accelerator, from_keras, load  # noqa: E402
from .errors import Onnx2FpgaError  # noqa: E402
from .hardware_layer import HardwareLayer  # noqa: E402

__all__ = ["Accelerator", "HardwareLayer", "Onnx2FpgaError", "errors", "from_keras",
           "load"]
