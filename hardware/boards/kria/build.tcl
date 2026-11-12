# A Kria bitstream in one batch run: vivado -mode batch -source build.tcl, from the directory onnx2fpga wrote.

source settings.tcl
if {![info exists ddr]} { set ddr 0 }
file delete -force $board.bit $board.bin $board.xsa

create_project -force $board vivado -part $part
if {$ddr} {
    add_files {ddr_shell.v ddr_top.v}
} else {
    add_files {kria_shell.v kria_top.v}
}
add_files -fileset constrs_1 kria.xdc
if {$sfp} {
    # The SFP+ pins wait, disabled, for the 10G MAC of spec 007 stage B.
    add_files -fileset constrs_1 sfp.xdc
    set_property IS_ENABLED false [get_files sfp.xdc]
}

create_bd_design system
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e ps]
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 1 \
    CONFIG.PSU__MAXIGP0__DATA_WIDTH 32 \
    CONFIG.PSU__USE__M_AXI_GP1 0 \
    CONFIG.PSU__USE__M_AXI_GP2 0 \
    CONFIG.PSU__FPGA_PL0_ENABLE 1 \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__SRCSEL IOPLL \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ 100 \
] $ps
if {$ddr} {
    set_property -dict [list CONFIG.PSU__USE__S_AXI_GP2 1 CONFIG.PSU__SAXIGP2__DATA_WIDTH 128] $ps
}
set smc [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect smc]
set_property -dict [list CONFIG.NUM_SI 1 CONFIG.NUM_MI 1] $smc

set m_axi [create_bd_intf_port -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI]
set_property -dict [list CONFIG.PROTOCOL AXI4LITE CONFIG.DATA_WIDTH 32 CONFIG.ADDR_WIDTH 32] $m_axi
create_bd_port -dir O -type clk pl_clk0
create_bd_port -dir O -type rst pl_resetn0
set aclk [create_bd_port -dir I -type clk -freq_hz 322265625 aclk]
create_bd_port -dir I -type rst aresetn
set_property -dict [list CONFIG.ASSOCIATED_BUSIF M_AXI CONFIG.ASSOCIATED_RESET aresetn] $aclk

connect_bd_intf_net [get_bd_intf_pins ps/M_AXI_HPM0_FPD] [get_bd_intf_pins smc/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins smc/M00_AXI] $m_axi
connect_bd_net $aclk [get_bd_pins ps/maxihpm0_fpd_aclk] [get_bd_pins smc/aclk]
connect_bd_net [get_bd_ports aresetn] [get_bd_pins smc/aresetn]
connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_ports pl_clk0]
connect_bd_net [get_bd_pins ps/pl_resetn0] [get_bd_ports pl_resetn0]
assign_bd_address -offset $base -range 64K \
    -target_address_space [get_bd_addr_spaces ps/Data] [get_bd_addr_segs M_AXI/Reg]
if {$ddr} {
    set memory [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect memory]
    set_property -dict [list CONFIG.NUM_SI 1 CONFIG.NUM_MI 1] $memory
    set port [create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 S_DDR]
    set_property -dict [list CONFIG.PROTOCOL AXI4 CONFIG.DATA_WIDTH 128 CONFIG.ADDR_WIDTH 40 \
        CONFIG.ID_WIDTH 1 CONFIG.READ_WRITE_MODE READ_ONLY] $port
    connect_bd_intf_net $port [get_bd_intf_pins memory/S00_AXI]
    connect_bd_intf_net [get_bd_intf_pins memory/M00_AXI] [get_bd_intf_pins ps/S_AXI_HP0_FPD]
    connect_bd_net $aclk [get_bd_pins memory/aclk] [get_bd_pins ps/saxihp0_fpd_aclk]
    connect_bd_net [get_bd_ports aresetn] [get_bd_pins memory/aresetn]
    set_property CONFIG.ASSOCIATED_BUSIF {M_AXI:S_DDR} $aclk
    assign_bd_address -offset 0x00000000 -range 2G \
        -target_address_space [get_bd_addr_spaces S_DDR] [get_bd_addr_segs ps/SAXIGP2/HP0_DDR_LOW]
    exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces S_DDR] \
        [get_bd_addr_segs ps/SAXIGP2/HP0_LPS_OCM]
}
validate_bd_design
save_bd_design
make_wrapper -files [get_files system.bd] -top -import
set_property top kria_top [current_fileset]

if {$ddr} {
    set_property strategy Performance_ExplorePostRoutePhysOpt [get_runs impl_1]
}
set_property STEPS.WRITE_BITSTREAM.ARGS.BIN_FILE true [get_runs impl_1]
launch_runs impl_1 -to_step write_bitstream -jobs $jobs
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
    error "implementation failed, see vivado/$board.runs/impl_1/runme.log"
}

open_run impl_1
report_timing_summary -max_paths 10 -file timing_summary.rpt
report_utilization -file utilization.rpt
source check_timing.tcl
set run [get_property DIRECTORY [get_runs impl_1]]
file copy -force [file join $run kria_top.bit] $board.bit
file copy -force [file join $run kria_top.bin] $board.bin
write_hw_platform -fixed -include_bit -force $board.xsa
