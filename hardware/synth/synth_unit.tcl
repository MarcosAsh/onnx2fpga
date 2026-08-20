# Out of context synthesis of one streaming unit.
#
# Driven entirely by environment variables so no argument quoting is involved:
#
#   OTF_PART     target part, for example xc7z020-clg400-1
#   OTF_TOP      module to synthesise
#   OTF_RTL      space separated SystemVerilog sources
#   OTF_GENERICS space separated NAME=VALUE overrides
#   OTF_PERIOD   target clock period in ns
#   OTF_OUT      directory to write reports and summary.json into
#   OTF_IMPL     1 to place and route as well, 0 for synthesis only
#
# Run it from the build directory of a compiled design, because the weight
# memories are initialised by $readmemh with paths relative to the working
# directory. Without them the memories read as undefined and synthesis
# optimises the datapath away, which would make every number here meaningless.

proc env_or {name fallback} {
    if {[info exists ::env($name)]} { return $::env($name) }
    return $fallback
}

set part     $::env(OTF_PART)
set top      $::env(OTF_TOP)
set sources  $::env(OTF_RTL)
set generics [env_or OTF_GENERICS ""]
set period   [env_or OTF_PERIOD 5.0]
set outdir   [env_or OTF_OUT "synth_out"]
set do_impl  [env_or OTF_IMPL 0]

file mkdir $outdir

foreach source $sources {
    read_verilog -sv $source
}

set xdc [file join $outdir clock.xdc]
set handle [open $xdc w]
puts $handle "create_clock -name clk -period $period \[get_ports clk\]"
close $handle
read_xdc -mode out_of_context $xdc

set overrides {}
foreach pair $generics {
    lappend overrides -generic $pair
}

synth_design -top $top -part $part -mode out_of_context {*}$overrides
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

# Counting primitives directly avoids parsing a human readable report whose
# layout changes between releases.
proc count_group {group} {
    return [llength [get_cells -hier -filter "PRIMITIVE_GROUP == $group"]]
}
proc count_ref {pattern} {
    return [llength [get_cells -hier -filter "REF_NAME =~ $pattern"]]
}

set lut    [count_group LUT]
set ff     [count_group FLOP_LATCH]
set dsp    [count_ref "DSP*"]
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
