"""onnx2fpga: an ONNX to streaming-dataflow FPGA compiler.

The pipeline runs strictly in stage order:

    p1_ingest    .onnx bytes            -> ONNX protos
    p2_graph     ONNX protos            -> compiler IR, channel-last
    p3_ops       operator library used by every later stage
    p4_quantize  float IR               -> integer hardware IR
    p5_schedule  hardware IR            -> folding factors, FIFOs, a balanced pipeline
    p6_emit      hardware IR            -> SystemVerilog, memory images, build files

targets holds the device models the scheduler budgets against; reference holds
the numpy golden model the generated RTL is checked against.
"""

__version__ = "0.1.0"
