# Keep the MMCM clock related to the PS clock, including its jitter and insertion delay.
create_generated_clock -name otf_clk [get_pins -of_objects [get_cells -hierarchical otf_mmcm] -filter {REF_PIN_NAME == CLKOUT0}]

set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]
