set clock [get_clocks -quiet otf_clk]
if {[llength $clock] != 1 || ![get_property IS_GENERATED $clock]} {
    error "otf_clk must be an MMCM generated clock"
}
set period [get_property PERIOD $clock]
if {abs($period - 1000.0 / 322.265625) > 0.0001} {
    error "unexpected otf_clk period: $period ns"
}

set checks {no_clock unconstrained_internal_endpoints generated_clocks loops}
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
