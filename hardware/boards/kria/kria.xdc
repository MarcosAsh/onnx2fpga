# 322.265625 MHz, the 10GBASE-R rate on a 32 bit datapath; otf_mmcm makes it from the 100 MHz pl_clk0.
create_clock -name otf_clk -period 3.103 [get_pins -of_objects [get_cells -hierarchical otf_mmcm] -filter {REF_PIN_NAME == CLKOUT0}]

set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]
