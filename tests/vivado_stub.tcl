# Stand-ins for the Vivado commands synth_unit.tcl and the Kria build.tcl call, so they run under plain tclsh.

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
proc get_ports {args} { return "clk" }

# The Kria project flow: a run directory with a bitstream in it, and a property for each question build.tcl asks.
set ::stub_run [file join [pwd] vivado stub_run]

proc get_property {property object} {
    switch $property {
        PROGRESS { return "100%" }
        DIRECTORY { return $::stub_run }
        default { return $::stub_slack }
    }
}

# Every file must exist, and constraints are sourced so the stubs check their syntax too.
proc add_files {args} {
    record add_files $args
    foreach item $args {
        if {[string match -* $item] || $item eq "constrs_1"} { continue }
        foreach path $item {
            if {![file exists $path]} { error "add_files: no file $path" }
            if {[file extension $path] eq ".xdc"} { source $path }
        }
    }
}

proc launch_runs {args} {
    record launch_runs $args
    file mkdir $::stub_run
    foreach name {kria_top.bit kria_top.bin} {
        set handle [open [file join $::stub_run $name] w]
        close $handle
    }
}

foreach command {
    create_project set_property get_files create_bd_design create_bd_cell
    create_bd_intf_port create_bd_port get_bd_intf_pins get_bd_pins get_bd_ports
    connect_bd_intf_net connect_bd_net assign_bd_address get_bd_addr_spaces
    get_bd_addr_segs validate_bd_design save_bd_design make_wrapper current_fileset
    get_runs wait_on_run open_run write_hw_platform create_clock get_pins current_design
} {
    proc $command {args} "record $command \$args; return $command"
}

# Called directly or through tests/fake_vivado, which passes the real tool's arguments.
set script [file join [file dirname [info script]] .. hardware synth synth_unit.tcl]
set index [lsearch $argv -source]
if {$index >= 0} { set script [lindex $argv [expr {$index + 1}]] }
source $script

set out [expr {[info exists ::env(OTF_OUT)] ? $::env(OTF_OUT) : "."}]
set handle [open [file join $out calls.txt] w]
foreach call $::calls { puts $handle $call }
close $handle
