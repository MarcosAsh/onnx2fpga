# KR260 SFP+ on GTH quad 224 lane 2 (GTHE4_CHANNEL_X1Y12), pins as in taxi's KR260 example sfp.xdc.
set_property PACKAGE_PIN T2 [get_ports sfp_rx_p]
set_property PACKAGE_PIN T1 [get_ports sfp_rx_n]
set_property PACKAGE_PIN R4 [get_ports sfp_tx_p]
set_property PACKAGE_PIN R3 [get_ports sfp_tx_n]

# The 156.25 MHz reference from U90 on MGTREFCLK0_224.
set_property PACKAGE_PIN Y6 [get_ports sfp_mgt_refclk_p]
set_property PACKAGE_PIN Y5 [get_ports sfp_mgt_refclk_n]
create_clock -name sfp_mgt_refclk -period 6.400 [get_ports sfp_mgt_refclk_p]
