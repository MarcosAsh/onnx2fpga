# Custom hardware layers

Status: partial (narrow scope, see Limits)

Sooner or later a model will use an operator we do not support, and `load` will
refuse it by name. You can fill the gap yourself by subclassing
`otf.HardwareLayer`, much like a Keras custom layer fills in `call`.

```python
class Relu6(otf.HardwareLayer):
    op_type = "Clip"
    module = "otf_act"

    def call(self, x):                  # float reference
        return np.clip(x, 0.0, 6.0)

    def call_integer(self, q):          # what the hardware computes, bit for bit
        ...

acc = otf.load("model.onnx", custom_objects={"Clip": Relu6})
```

`call` is the float behaviour, which calibration and `evaluate` use.
`call_integer` is the integer behaviour, and it becomes the golden model the
RTL is checked against, so get it exactly right. `module` and `parameters(pe)`
name the hardware unit and its parameters. `memories(pe)` and
`resources(device, pe)` are optional.

There are two worked examples in `tests/test_api.py`. One maps ONNX `Clip`,
which is what Keras's ReLU6 exports as, onto our existing `otf_act` unit. The
other implements `Neg` with its own small SystemVerilog module.

## Limits

A custom layer gets one stream in and one out, keeps the shape and the integer
grid, and folds by `pe`. That covers elementwise operators and nothing bigger.
Softmax, for example, reduces across channels and needs a new output scale,
which means changes to the quantizer and the scheduler. It cannot be a plug in
yet.

On the Hardcaml backend the module also has to exist in
`hardware/hardcaml/src/units.ml` already. Name one that does not and you get a
`ConfigError` that says so.

## Open questions

Should a custom layer be able to ship its own Hardcaml module and register it
at run time, instead of somebody editing `units.ml`?
