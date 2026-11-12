# Pins from Digilent's Arty-A7-35/100-Master.xdc, revisions D and E.
create_clock -name sys_clk -period 10 [get_ports clk]
set_property -dict {PACKAGE_PIN E3 IOSTANDARD LVCMOS33} [get_ports {clk}]
set_property -dict {PACKAGE_PIN D9 IOSTANDARD LVCMOS33} [get_ports {button}]
set_property -dict {PACKAGE_PIN A9 IOSTANDARD LVCMOS33} [get_ports {rxd}]
set_property -dict {PACKAGE_PIN D10 IOSTANDARD LVCMOS33} [get_ports {txd}]
set_property -dict {PACKAGE_PIN H5 IOSTANDARD LVCMOS33} [get_ports {led[0]}]
set_property -dict {PACKAGE_PIN J5 IOSTANDARD LVCMOS33} [get_ports {led[1]}]
# Button and UART inputs enter synchronizers; UART and LEDs have no external clock.
set_false_path -from [get_ports {button rxd}]
set_false_path -to [get_ports {txd led[*]}]
