"""Every exception the compiler raises on purpose, deriving from the builtin a caller would catch."""


class Onnx2FpgaError(Exception):
    pass


class ModelError(Onnx2FpgaError, ValueError):
    pass


class UnsupportedOpError(ModelError):
    def __init__(self, op_type, node, supported):
        self.op_type = op_type
        self.node = node
        self.supported = tuple(supported)
        ModelError.__init__(
            self, "unsupported op %r in node %r (supported: %s); add it with "
            "custom_objects={%r: YourHardwareLayer}"
            % (op_type, node, ", ".join(self.supported), op_type))


class ConfigError(Onnx2FpgaError, ValueError):
    pass


class UnknownLayerError(ConfigError, KeyError):
    """A KeyError that prints like any other error instead of as a quoted key."""

    __str__ = Exception.__str__


class BudgetError(Onnx2FpgaError):
    pass


class ToolchainError(Onnx2FpgaError, RuntimeError):
    def __init__(self, message, tool=None, remedy=None):
        self.tool = tool
        self.remedy = remedy
        if remedy:
            message = "%s\nTo fix: %s" % (message, remedy)
        Onnx2FpgaError.__init__(self, message)


class NotCompiledError(Onnx2FpgaError, RuntimeError):
    pass


class NotDeployedError(Onnx2FpgaError, RuntimeError):
    pass


class SimulationError(Onnx2FpgaError, RuntimeError):
    def __init__(self, message, stdout="", stderr=""):
        self.stdout = stdout
        self.stderr = stderr
        Onnx2FpgaError.__init__(self, message)
