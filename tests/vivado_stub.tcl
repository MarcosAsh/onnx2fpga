# Stand-ins for the Vivado commands synth_unit.tcl calls.
#
# There is no Vivado on the development machine, so the script that drives it
# would otherwise ship completely unexecuted. Sourcing it under plain tclsh
# with these stubs still checks its control flow, its arithmetic, and the exact
# shape of the summary it writes, which is the interface the Python side reads.
# What it cannot check is whether Vivado agrees, so the first real run is still
# the one that matters.

set ::calls {}

# Canned primitive counts, keyed by the filter the script uses.
array set ::primitives {
    "PRIMITIVE_GROUP == LUT"         412
    "PRIMITIVE_GROUP == FLOP_LATCH"  288
    "REF_NAME =~ DSP*"               6
    "REF_NAME =~ RAMB36*"            1
    "REF_NAME =~ RAMB18*"            2
    "REF_NAME =~ URAM*"              0
}
set ::stub_slack 0.842

proc record {name arguments} {
    lappend ::calls [concat $name $arguments]
}

proc read_verilog {args} { record read_verilog $args }
proc read_xdc {args} { record read_xdc $args }
proc synth_design {args} { record synth_design $args }
proc opt_design {args} { record opt_design $args }
proc place_design {args} { record place_design $args }
proc phys_opt_design {args} { record phys_opt_design $args }
proc route_design {args} { record route_design $args }

proc write_stub_report {arguments} {
    set index [lsearch $arguments -file]
    if {$index >= 0} {
        set path [lindex $arguments [expr {$index + 1}]]
        file mkdir [file dirname $path]
        set handle [open $path w]
        puts $handle "stub report"
        close $handle
    }
}

proc report_utilization {args} { record report_utilization $args; write_stub_report $args }
proc report_timing_summary {args} { record report_timing_summary $args; write_stub_report $args }

proc get_cells {args} {
    record get_cells $args
    set index [lsearch $args -filter]
    set filter [lindex $args [expr {$index + 1}]]
    set count 0
    if {[info exists ::primitives($filter)]} { set count $::primitives($filter) }
    set cells {}
    for {set i 0} {$i < $count} {incr i} { lappend cells "cell_$i" }
    return $cells
}

proc get_timing_paths {args} { record get_timing_paths $args; return "worst_path" }
proc get_property {property object} { return $::stub_slack }
proc get_ports {args} { return "clk" }

# Invoked either directly or through tests/fake_vivado, which passes the real
# tool's argument list.
set script [file join [file dirname [info script]] .. hardware synth synth_unit.tcl]
set index [lsearch $argv -source]
if {$index >= 0} { set script [lindex $argv [expr {$index + 1}]] }
source $script

set handle [open [file join $::env(OTF_OUT) calls.txt] w]
foreach call $::calls { puts $handle $call }
close $handle
