"""What quantization cost, measured rather than assumed.

Every number a compiled design produces is an approximation of the float model
it came from, and until now the compiler said nothing about how good an
approximation. That is the wrong silence: the error is knowable at compile
time, from the samples already used for calibration, and it is knowable before
anyone spends an hour on synthesis finding out the accuracy was never going to
be acceptable.

Three figures, because no one of them answers the question on its own.

The absolute error is what a user's own tolerance is written in. The relative
error, against the float model's own range, is what says whether the compiled
model is a good approximation, and it is the one a wider datapath improves.

Steps are the subtle one. An error measured in output quantization steps says
how far the integer pipeline lands from the float model in units of its own
grid, and it stays roughly constant as the datapath widens, because widening
shrinks the step by about as much as it shrinks the error. On the example MLP
int8 and int16 both sit near two and a half steps while the absolute error
falls by two hundred and fifty times. So steps are the wrong thing to chase
and a good thing to know: they say the pipeline is tracking its grid, not that
the grid is fine enough.
"""

import numpy as np

from .reference.runner import GraphRunner


class OutputAccuracy:
    """One graph output, compared across every sample."""

    def __init__(self, name, worst, mean, step, reference_peak):
        self.name = name
        self.worst = worst
        self.mean = mean
        self.step = step
        self.reference_peak = reference_peak

    @property
    def worst_steps(self):
        return self.worst / self.step if self.step else float("inf")

    @property
    def mean_steps(self):
        return self.mean / self.step if self.step else float("inf")

    @property
    def relative(self):
        """Worst error against the float model's own range, which is the form
        that survives being quoted without the scale beside it."""
        return self.worst / self.reference_peak if self.reference_peak else 0.0

    #: Above this share of the float model's own range, the approximation is
    #: loose enough to be worth widening the datapath for. Below it, the
    #: remaining error is mostly the grid and a wider one buys little.
    LOOSE = 0.01

    @property
    def is_loose(self):
        return self.relative > self.LOOSE

    @property
    def tracks_its_grid(self):
        """Within a few steps means the integer pipeline is doing what the
        grid allows. It does not mean the grid is fine enough; that is what
        the relative figure is for."""
        return self.worst_steps <= 3.0


class AccuracyReport:
    def __init__(self, rows, samples):
        self.rows = rows
        self.samples = samples

    @classmethod
    def measure(cls, float_graph, hardware_graph, plan, samples, limit=32):
        """Run both graphs over the same inputs and difference the outputs.

        The float graph is the model as written; the hardware graph is exactly
        what the RTL will do, since it is the same golden model the Verilator
        harness is checked against. So this is not a model of the error, it is
        the error, on these samples.
        """
        if not samples:
            return None
        source = hardware_graph.inputs[0]
        spec = plan.spec(source)
        worst = {}
        total = {}
        peak = {}
        used = 0
        for feeds in samples[:limit]:
            values = np.asarray(list(feeds.values())[0], dtype=np.float64)
            reference = GraphRunner(float_graph).run(
                {float_graph.inputs[0]: values})
            integers = spec.quantize(values, plan.act_dtype)
            produced = GraphRunner(hardware_graph).run({source: integers})
            used += 1
            for sink in hardware_graph.outputs:
                expected = np.asarray(reference[sink], dtype=np.float64)
                actual = ((np.asarray(produced[sink], dtype=np.float64)
                           - plan.zero_point(sink)) * plan.scale(sink))
                error = np.abs(actual - expected)
                worst[sink] = max(worst.get(sink, 0.0), float(np.max(error)))
                total[sink] = total.get(sink, 0.0) + float(np.mean(error))
                peak[sink] = max(peak.get(sink, 0.0),
                                 float(np.max(np.abs(expected))))
        rows = [OutputAccuracy(sink, worst[sink], total[sink] / used,
                               plan.scale(sink), peak[sink])
                for sink in hardware_graph.outputs]
        return cls(rows, used)

    @property
    def worst(self):
        return max(self.rows, key=lambda row: row.relative)

    def render(self):
        lines = ["accuracy against the float model, over %d sample%s"
                 % (self.samples, "" if self.samples == 1 else "s"),
                 "%-20s %12s %12s %10s %10s" %
                 ("output", "worst", "worst/step", "mean/step", "relative")]
        for row in self.rows:
            lines.append("%-20s %12.3e %12.2f %10.2f %9.3f%%%s" % (
                row.name, row.worst, row.worst_steps, row.mean_steps,
                100 * row.relative, "  <-- loose" if row.is_loose else ""))
        worst = self.worst
        if worst.is_loose:
            lines.append(
                "worst output is %.2f%% of the float model's own range; widen "
                "the datapath (--act-bits 16 --weight-bits 16) if that is too "
                "much for you" % (100 * worst.relative))
        else:
            lines.append(
                "worst output is %.3f%% of the float model's own range"
                % (100 * worst.relative))
        if not worst.tracks_its_grid:
            lines.append(
                "note %s is %.1f steps from the float model, which is more than "
                "rounding: widening will not fix a discrepancy of that shape"
                % (worst.name, worst.worst_steps))
        return "\n".join(lines)
