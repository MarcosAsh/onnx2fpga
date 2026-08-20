"""Pass infrastructure."""

import time


class CompilerContext:
    def __init__(self, device, config=None):
        self.device = device
        self.config = dict(config or {})
        self.artifacts = {}
        self.log = []
        self.timings = {}

    def note(self, message):
        self.log.append(message)
        return message

    def option(self, key, default=None):
        return self.config.get(key, default)


class Pass:
    name = "pass"

    def run(self, graph, context):
        raise NotImplementedError

    def __call__(self, graph, context):
        return self.run(graph, context)


class PassManager:
    def __init__(self, passes, verbose=False):
        self.passes = list(passes)
        self.verbose = verbose

    def run(self, graph, context):
        """Timings are wall clock, so they are kept beside the log rather than
        inside it. Everything the compiler writes down has to be reproducible,
        or two runs of the same model produce different files."""
        for stage in self.passes:
            started = time.perf_counter()
            graph = stage(graph, context) or graph
            graph.validate()
            context.timings[stage.name] = (time.perf_counter() - started) * 1e3
            context.note("%-22s %3d nodes" % (stage.name, len(graph.nodes)))
            if self.verbose:
                print("%s  %6.1f ms" % (context.log[-1],
                                        context.timings[stage.name]))
        return graph
