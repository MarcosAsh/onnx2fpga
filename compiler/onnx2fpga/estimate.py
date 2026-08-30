"""Will it fit, and how fast, without committing to a build.

Compiling a model writes RTL, memory images, golden vectors and a build
system, and then synthesis takes an hour. Both of those are a poor way to find
out that a design needed twice the DSPs on the device. Every number needed to
answer the question is known once folding has run, which is well before
anything is written, so the question can be answered from the .onnx alone.

The frame figures come from the same latency model the harness is checked
against, so they are the compiler's real prediction rather than a separate
guess made for reporting. The nanoseconds carry the device's assumed clock and
say so, because that assumption is the weakest number here until a synthesis
run replaces it.
"""

from .p5_schedule.latency import LatencyModel


class Estimate:
    def __init__(self, graph, folding, device):
        self.graph = graph
        self.folding = folding
        self.device = device
        self.latency = LatencyModel(graph)

    @property
    def latency_cycles(self):
        return self.latency.cycles

    @property
    def frame_cycles(self):
        return self.folding.bottleneck_cycles if self.folding else 0

    def _nanoseconds(self, cycles):
        if not self.device.fmax_mhz:
            return None
        return 1000.0 * cycles / self.device.fmax_mhz

    @property
    def latency_ns(self):
        return self._nanoseconds(self.latency_cycles)

    @property
    def utilisation(self):
        if not self.folding:
            return {}
        return self.folding.total.utilisation(self.device.budget)

    @property
    def worst_utilisation(self):
        used = self.utilisation
        return max(used.values()) if used else 0.0

    @property
    def fits(self):
        return self.worst_utilisation <= 1.0

    @property
    def tightest(self):
        """Which resource is closest to running out, since that is the one
        that decides whether a bigger model still fits."""
        used = self.utilisation
        if not used:
            return None
        return max(used, key=used.get)

    def render(self):
        verdict = "fits" if self.fits else "DOES NOT FIT"
        lines = ["%s on %s, %d cycles to first output"
                 % (verdict, self.device.name, self.latency_cycles)]
        if self.latency_ns is not None:
            lines[0] += " (%.0f ns at an assumed %.0f MHz)" % (
                self.latency_ns, self.device.fmax_mhz)
        if self.folding:
            lines.append("%d cycles per frame at the slowest unit"
                         % self.frame_cycles)
            lines.append("  ".join(
                "%s %.1f%%" % (field, 100 * value)
                for field, value in sorted(self.utilisation.items())))
            if self.tightest:
                lines.append("tightest is %s at %.1f%% of %s"
                             % (self.tightest,
                                100 * self.utilisation[self.tightest],
                                self.device.name))
        if not self.fits:
            lines.append("over budget: fold it harder with --target-cycles, "
                         "raise --utilisation, or pick a larger device")
        lines.append("resource figures are the estimator's, not synthesis; "
                     "the clock is assumed, not measured")
        return "\n".join(lines)
