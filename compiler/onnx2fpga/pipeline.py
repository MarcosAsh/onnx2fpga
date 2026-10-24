"""Pass infrastructure."""

import time


class CompilerContext:
    def __init__(self, device):
        self.device = device
        self.artifacts = {}
        self.log = []
        self.timings = {}

    def note(self, message):
        self.log.append(message)


class Pass:
    name = "pass"

    def run(self, graph, context):
        raise NotImplementedError


class PassManager:
    def __init__(self, passes, verbose=False):
        self.passes = list(passes)
        self.verbose = verbose

    def run(self, graph, context):
        for stage in self.passes:
            started = time.perf_counter()
            first = len(context.log)
            graph = stage.run(graph, context) or graph
            graph.validate()
            # Timings stay out of the log so the log is the same on every run.
            context.timings[stage.name] = (time.perf_counter() - started) * 1e3
            context.note("%-22s %3d nodes" % (stage.name, len(graph.nodes)))
            if self.verbose:
                for message in context.log[first:-1]:
                    print(message)
                print("%s  %6.1f ms" % (context.log[-1], context.timings[stage.name]))
        return graph
