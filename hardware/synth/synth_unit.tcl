# Out of context synthesis of one unit, configured by OTF_* env vars; run it from the design build dir so $readmemh finds the weights.

proc env_or {name fallback} {
    if {[info exists ::env($name)]} { return $::env($name) }
    return $fallback
}

set part     $::env(OTF_PART)
set top      $::env(OTF_TOP)
set sources  [env_or OTF_RTL ""]
set generics [env_or OTF_GENERICS ""]
set period   [env_or OTF_PERIOD 5.0]
set outdir   [env_or OTF_OUT "synth_out"]
set do_impl  [env_or OTF_IMPL 0]
set clock    [env_or OTF_CLOCK clk]
set checkpoint [env_or OTF_CHECKPOINT ""]
set replicate_net [env_or OTF_REPLICATE_NET ""]

if {$replicate_net ne "" && $checkpoint eq ""} { error "replication requires a checkpoint" }

file mkdir $outdir

if {$checkpoint ne ""} {
    open_checkpoint $checkpoint
    if {[get_property PART [current_design]] ne $part} { error "checkpoint part mismatch" }
    if {[get_property PERIOD [get_clocks clk]] != $period} { error "checkpoint period mismatch" }
    if {$replicate_net ne ""} {
        set net [get_nets -quiet $replicate_net]
        if {[llength $net] != 1 || [get_property NAME $net] ne $replicate_net} {
            error "replication requires one exact net name: $replicate_net"
        }
        route_design -unroute
        phys_opt_design -force_replication_on_nets $net
    } else {
        phys_opt_design -directive AggressiveExplore
    }
    route_design -directive Explore
    report_utilization -file [file join $outdir utilization_impl.rpt]
    report_timing_summary -file [file join $outdir timing_impl.rpt]
    set stage impl
} else {
    foreach source $sources {
        read_verilog -sv $source
    }

    set xdc [file join $outdir clock.xdc]
    set handle [open $xdc w]
    puts $handle "create_clock -name clk -period $period \[get_ports [list $clock]\]"
    close $handle
    read_xdc -mode out_of_context $xdc

    set overrides {}
    foreach pair $generics {
        lappend overrides -generic $pair
    }

    synth_design -top $top -part $part -mode out_of_context {*}$overrides
    if {[llength [get_clocks -quiet clk]] != 1} {
        error "missing synthesis clock on port $clock"
    }
    report_utilization -file [file join $outdir utilization_synth.rpt]
    report_timing_summary -file [file join $outdir timing_synth.rpt]
    set stage synth

    if {$do_impl} {
        opt_design
        place_design
        phys_opt_design
        route_design
        report_utilization -file [file join $outdir utilization_impl.rpt]
        report_timing_summary -file [file join $outdir timing_impl.rpt]
        set stage impl
    }
}

proc report_count {report labels} {
    foreach line [split $report "\n"] {
        set fields [split $line |]
        set label [string trim [lindex $fields 1] " *"]
        if {$label in $labels} {
            set count [string map {, ""} [string trim [lindex $fields 2]]]
            if {[string is integer -strict $count] && $count >= 0} { return $count }
            error "invalid utilization count for $label"
        }
    }
    error "missing utilization row: $labels"
}

proc count_ref {pattern} {
    return [llength [get_cells -hier -filter "REF_NAME =~ $pattern"]]
}

set handle [open [file join $outdir utilization_${stage}.rpt] r]
set utilization [read $handle]
close $handle
set lut [report_count $utilization {"CLB LUTs" "Slice LUTs"}]
set ff  [report_count $utilization {"CLB Registers" "Slice Registers"}]
set dsp    [report_count $utilization {DSPs DSP48E1 DSP48E2}]
set ramb36 [count_ref "RAMB36*"]
set ramb18 [count_ref "RAMB18*"]
set uram   [count_ref "URAM*"]

set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
if {$wns eq ""} { set wns 0 }
set achieved [expr {$period - $wns}]
if {$achieved <= 0} { set fmax 0 } else { set fmax [expr {1000.0 / $achieved}] }

set handle [open [file join $outdir summary.json] w]
puts $handle "{"
puts $handle "  \"top\": \"$top\","
puts $handle "  \"part\": \"$part\","
puts $handle "  \"stage\": \"$stage\","
puts $handle "  \"generics\": \"$generics\","
puts $handle "  \"period_ns\": $period,"
puts $handle "  \"wns_ns\": $wns,"
puts $handle "  \"fmax_mhz\": [format %.2f $fmax],"
puts $handle "  \"lut\": $lut,"
puts $handle "  \"ff\": $ff,"
puts $handle "  \"dsp\": $dsp,"
puts $handle "  \"ramb36\": $ramb36,"
puts $handle "  \"ramb18\": $ramb18,"
puts $handle "  \"bram36\": [expr {$ramb36 + 0.5 * $ramb18}],"
puts $handle "  \"uram\": $uram"
puts $handle "}"
close $handle

puts "OTF_SUMMARY [file join $outdir summary.json]"

if {$stage eq "impl"} { write_checkpoint -force [file join $outdir routed.dcp] }
