open! Core
open! Hardcaml
open! Signal

module type Config = sig
  val clocks_per_bit : int
end

module Make (Config : Config) = struct
  open Config

  let () =
    if clocks_per_bit < 4
    then
      raise_s
        [%message
          "BUG: uart_rx samples the middle of each bit, so needs four clocks a bit"
            (clocks_per_bit : int)]
  ;;

  (* A character is low for nine bits at most, so twice that is a break. *)
  let break_clocks = 20 * clocks_per_bit
  let timer_bits = num_bits_to_represent (clocks_per_bit - 1)
  let run_bits = num_bits_to_represent break_clocks

  module I = struct
    type 'a t =
      { clocking : 'a Clocking.t
      ; rxd : 'a
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t =
      { byte : 'a With_valid.t [@bits 8]
      ; framing_error : 'a
      ; line_break : 'a
      }
    [@@deriving hardcaml]
  end

  module State = struct
    type t =
      | Idle
      | Start
      | Data
      | Stop
    [@@deriving sexp_of, compare ~localize, enumerate]
  end

  let create (scope : Scope.t) (i : Signal.t I.t) =
    let spec = Clocking.to_spec i.clocking in
    let%hw rxd_meta = reg spec ~clear_to:vdd i.rxd in
    let%hw rxd = reg spec ~clear_to:vdd rxd_meta in
    let%hw rxd_last = reg spec ~clear_to:vdd rxd in
    let%hw falling = rxd_last &: ~:rxd in
    let%hw low_run =
      reg_fb spec ~width:run_bits ~f:(fun run ->
        mux2 rxd (zero run_bits) @@ mux2 (run ==:. break_clocks) run (run +:. 1))
    in
    let%hw line_break = low_run ==:. break_clocks in
    let%hw.Always.State_machine sm = Always.State_machine.create (module State) spec in
    let%hw_var timer = Always.Variable.reg spec ~width:timer_bits in
    let%hw_var bit_index = Always.Variable.reg spec ~width:3 in
    let%hw_var shift = Always.Variable.reg spec ~width:8 in
    let%hw_var valid = Always.Variable.reg spec ~width:1 in
    let%hw_var framing_error = Always.Variable.reg spec ~width:1 in
    let%hw sample_now = timer.value ==:. 0 in
    let wait = Always.(timer <-- timer.value -:. 1) in
    let next_bit = Always.(timer <--. clocks_per_bit - 1) in
    Always.(
      compile
        [ valid <--. 0
        ; framing_error <--. 0
        ; sm.switch
            [ ( Idle
              , [ when_ falling [ timer <--. (clocks_per_bit / 2) - 1; sm.set_next Start ]
                ] )
            ; ( Start
              , [ if_
                    sample_now
                    [ if_
                        rxd
                        [ sm.set_next Idle ]
                        [ next_bit; bit_index <--. 0; sm.set_next Data ]
                    ]
                    [ wait ]
                ] )
            ; ( Data
              , [ if_
                    sample_now
                    [ next_bit
                    ; shift <-- rxd @: shift.value.:[7, 1]
                    ; bit_index <-- bit_index.value +:. 1
                    ; when_ (bit_index.value ==:. 7) [ sm.set_next Stop ]
                    ]
                    [ wait ]
                ] )
            ; ( Stop
              , [ if_
                    sample_now
                    [ valid <-- rxd; framing_error <-- ~:rxd; sm.set_next Idle ]
                    [ wait ]
                ] )
            ]
        ]);
    { O.byte = { valid = valid.value; value = shift.value }
    ; framing_error = framing_error.value
    ; line_break
    }
  ;;

  let hierarchical ?instance scope i =
    let module H = Hierarchy.In_scope (I) (O) in
    H.hierarchical ?instance ~scope ~name:"uart_rx" create i
  ;;
end
