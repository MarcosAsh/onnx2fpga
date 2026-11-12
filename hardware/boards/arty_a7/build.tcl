source settings.tcl
file delete -force $board.bit
read_verilog arty_top.v
read_xdc arty.xdc
synth_design -top arty_top -part $part
set rx_sync [get_cells -hier -filter {NAME =~ *rxd_meta_reg || NAME =~ *rxd_0_reg}]
if {[llength $rx_sync] != 2} { error "missing UART synchronizer" }
set_property ASYNC_REG TRUE $rx_sync
opt_design
place_design
phys_opt_design
route_design
report_timing_summary -file timing_summary.rpt
report_utilization -file utilization.rpt
source check_timing.tcl
write_checkpoint -force routed.dcp
write_bitstream -force $board.bit
