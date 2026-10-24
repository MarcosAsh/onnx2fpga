# Errors

Status: implemented

Every error the API raises derives from `otf.Onnx2FpgaError` and says what
happened, what was expected and what to do.

| Error | Raised when | Example message |
|---|---|---|
| `ModelError` | The ONNX file is malformed or uses something unsupported. | |
| `UnsupportedOpError` | An operator has no handler. | names the operator, the node and the supported list |
| `ConfigError` | An argument or pin is invalid. | `fc0 cannot fold to pe=3. Valid pe values for it: 1, 2, 4, 8, 16, 32, 64 (pe divides the 64 outputs)` |
| `UnknownLayerError` | `acc.layers[name]` names no layer. Also a `KeyError`. | `the model has no layer named 'nope'. Layers: fc0, fc1, fc2` |
| `BudgetError` | The target cannot fit the device. | quotes the allocator's reason line and suggests what to change |
| `ToolchainError` | A tool a step needs is missing. | names the tool and how to install it |
| `NotCompiledError` | A step needs `compile()` first. | `predict needs a compiled design: call acc.compile(device=...) first` |
| `NotDeployedError` | `predict(on="board")` before `deploy()`. | |
| `SimulationError` | Verilator ran but the outputs did not match. | |

`BoardError` and `HardcamlError` also derive from `Onnx2FpgaError`.

From the command line, a missing tool exits with status 2 and a failed check
with status 1.
