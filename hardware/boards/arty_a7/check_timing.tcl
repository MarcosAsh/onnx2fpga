set clock [get_clocks -quiet sys_clk]
if {[llength [get_clocks]] != 1 || [llength $clock] != 1} {
    error "expected one sys_clk"
}
if {abs([get_property PERIOD $clock] - 10.0) > 0.0001} {
    error "expected 100 MHz sys_clk"
}

set checks {no_clock unconstrained_internal_endpoints generated_clocks loops multiple_clock latch_loops}
set checked [check_timing -override_defaults $checks -return_string]
foreach check $checks {
    set pattern [format {checking %s \(([0-9]+)\)} $check]
    if {![regexp $pattern $checked -> count] || $count != 0} {
        error "timing check failed: $check"
    }
}

report_methodology -file methodology.rpt
set violations [get_methodology_violations -quiet -filter {SEVERITY == {Critical Warning} || SEVERITY == Error}]
if {[llength $violations]} { error "critical methodology violations: $violations" }

set handle [open timing_summary.rpt r]
set report [read $handle]
close $handle
set summary {}
foreach line [split $report "\n"] {
    set fields [regexp -all -inline {\S+} $line]
    if {[llength $fields] != 12} { continue }
    set numeric 1
    foreach field $fields {
        if {![string is double -strict $field]} { set numeric 0; break }
    }
    if {$numeric} { set summary $fields; break }
}
if {[llength $summary] != 12} { error "missing design timing summary" }
foreach offset {0 4 8} kind {setup hold pulse_width} {
    lassign [lrange $summary $offset [expr {$offset + 3}]] slack total failing endpoints
    if {$slack < 0 || $total < 0 || $failing != 0 || $endpoints <= 0} {
        error "$kind timing failed: $slack ns, $failing failing endpoints"
    }
}
