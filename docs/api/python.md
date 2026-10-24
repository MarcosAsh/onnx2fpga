# Classes and functions

Generated from the docstrings in `compiler/onnx2fpga`. The
[guide to the API](reference.md) explains the same objects with examples.

```{eval-rst}
.. currentmodule:: onnx2fpga

.. autosummary::
   :nosignatures:

   load
   from_keras
   Accelerator
   HardwareLayer
   Onnx2FpgaError

.. autofunction:: load

.. autofunction:: from_keras

.. autoclass:: Accelerator
   :members: compile, summary, evaluate, predict, simulate, build, deploy, disconnect, layers, device, hdl, out_dir, deployed, input_name, input_shape

.. autoclass:: onnx2fpga.api.Layer
   :members: fold, quantize, folding, cycles, units

.. autoclass:: onnx2fpga.api.SimulationReport
   :members: ok, cycles, latency

.. autoclass:: HardwareLayer
   :members:

.. automodule:: onnx2fpga.errors
   :members:
   :show-inheritance:
```
