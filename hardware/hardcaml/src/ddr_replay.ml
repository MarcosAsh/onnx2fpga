open! Core
open! Hardcaml
open! Signal

module State = struct
  type t =
    | Idle
    | Address
    | Data
  [@@deriving enumerate, sexp_of, compare ~localize]
end

module I = struct
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; start : 'a
    ; abort_request : 'a
    ; base : 'a [@bits 40]
    ; records : 'a [@bits 32]
    ; ready : 'a
    ; arready : 'a
    ; rvalid : 'a
    ; rdata : 'a [@bits 128]
    ; rresp : 'a [@bits 2]
    ; rlast : 'a
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t =
    { arvalid : 'a
    ; araddr : 'a [@bits 40]
    ; arlen : 'a [@bits 8]
    ; arsize : 'a [@bits 3]
    ; arburst : 'a [@bits 2]
    ; rready : 'a
    ; data : 'a [@bits 64]
    ; keep : 'a [@bits 8]
    ; valid : 'a
    ; last : 'a
    ; busy : 'a
    ; done_ : 'a
    ; error : 'a
    ; aborted : 'a
    ; delivered : 'a [@bits 32]
    }
  [@@deriving hardcaml]
end

let create (scope : Scope.t) (i : Signal.t I.t) =
  let spec = Reg_spec.create ~clock:i.clock ~clear:i.clear () in
  let%hw.Always.State_machine sm = Always.State_machine.create (module State) spec in
  let%hw_var address = Always.Variable.reg spec ~width:40 in
  let%hw_var remaining = Always.Variable.reg spec ~width:32 in
  let%hw_var beats = Always.Variable.reg spec ~width:5 in
  let%hw_var drain = Always.Variable.reg spec ~width:1 in
  let%hw_var done_ = Always.Variable.reg spec ~width:1 in
  let%hw_var error = Always.Variable.reg spec ~width:1 in
  let%hw_var aborted = Always.Variable.reg spec ~width:1 in
  let%hw_var delivered = Always.Variable.reg spec ~width:32 in
  let%hw burst =
    mux2
      (remaining.value >:. 16)
      (of_int_trunc ~width:5 16)
      (sel_bottom remaining.value ~width:5)
  in
  let%hw arvalid = sm.is Address &: ~:(i.clear) in
  let%hw rready =
    sm.is Data &: (i.ready |: drain.value |: i.abort_request) &: ~:(i.clear)
  in
  let%hw read = rready &: i.rvalid in
  let%hw keep = i.rdata.:[71, 64] in
  let%hw last = i.rdata.:(72) in
  let%hw keep_ok =
    keep <>:. 0 &: (keep &: keep +:. 1 ==:. 0) &: (last |: (keep ==:. 255))
  in
  let%hw final_record = remaining.value ==:. 0 &: (beats.value ==:. 1) in
  let%hw record_ok = keep_ok &: (i.rdata.:[127, 73] ==:. 0) &: (~:final_record |: last) in
  let%hw response_ok = i.rresp ==:. 0 &: (i.rlast ==: (beats.value ==:. 1)) in
  let%hw valid =
    read &: ~:(drain.value) &: ~:(i.abort_request) &: record_ok &: response_ok
  in
  let%hw ending = uresize i.base ~width:41 +: sll (uresize i.records ~width:41) ~by:4 in
  let%hw config_ok = i.base.:[7, 0] ==:. 0 &: ~:(ending.:(40)) in
  let finish = Always.[ done_ <--. 1; sm.set_next Idle ] in
  Always.(
    compile
      [ when_ (~:(sm.is Idle) &: i.abort_request) [ drain <--. 1; aborted <--. 1 ]
      ; sm.switch
          [ ( Idle
            , [ when_
                  (i.start &: i.ready &: ~:(i.abort_request))
                  [ address <-- i.base
                  ; remaining <-- i.records
                  ; delivered <--. 0
                  ; drain <--. 0
                  ; done_ <--. 0
                  ; error <--. 0
                  ; aborted <--. 0
                  ; if_
                      ~:config_ok
                      ([ error <--. 1 ] @ finish)
                      [ if_ (i.records ==:. 0) finish [ sm.set_next Address ] ]
                  ]
              ] )
          ; ( Address
            , [ when_
                  i.arready
                  [ beats <-- burst
                  ; address <-- address.value +: sll (uresize burst ~width:40) ~by:4
                  ; remaining <-- remaining.value -: uresize burst ~width:32
                  ; sm.set_next Data
                  ]
              ] )
          ; ( Data
            , [ when_
                  read
                  [ when_ valid [ delivered <-- delivered.value +:. 1 ]
                  ; when_
                      (~:response_ok |: (~:record_ok &: ~:(drain.value)))
                      [ error <--. 1; drain <--. 1 ]
                  ; if_
                      i.rlast
                      [ if_
                          (drain.value
                           |: i.abort_request
                           |: ~:response_ok
                           |: ~:record_ok
                           |: (remaining.value ==:. 0))
                          finish
                          [ sm.set_next Address ]
                      ]
                      [ when_ (beats.value <>:. 0) [ beats <-- beats.value -:. 1 ] ]
                  ]
              ] )
          ]
      ]);
  { O.arvalid
  ; araddr = address.value
  ; arlen = uresize (burst -:. 1) ~width:8
  ; arsize = of_int_trunc ~width:3 4
  ; arburst = of_int_trunc ~width:2 1
  ; rready
  ; data = i.rdata.:[63, 0]
  ; keep
  ; valid
  ; last
  ; busy = ~:(sm.is Idle)
  ; done_ = done_.value
  ; error = error.value
  ; aborted = aborted.value
  ; delivered = delivered.value
  }
;;

let hierarchical ?instance scope i =
  let module H = Hierarchy.In_scope (I) (O) in
  H.hierarchical ?instance ~scope ~name:"ddr_replay" create i
;;
