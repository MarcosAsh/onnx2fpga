open! Core
open! Hardcaml
open! Signal

module type Config = sig
  val design : Design.t
end

module Make (Config : Config) = struct
  module Accelerator = Fixed_top.Make (Config)

  let () =
    if Accelerator.reloadable
    then raise_s [%message "fixed_stream requires frozen weights"]
  ;;

  module I = struct
    type 'a t =
      { clk : 'a
      ; rst_n : 'a
      ; s_tdata : 'a [@bits Design.width Config.design Config.design.input]
      ; s_tvalid : 'a
      ; m_tready : 'a
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t =
      { s_tready : 'a
      ; m_tdata : 'a [@bits Design.width Config.design Config.design.output]
      ; m_tvalid : 'a
      }
    [@@deriving hardcaml]
  end

  let create (scope : Scope.t) (i : Signal.t I.t) =
    let%hw clear = ~:(i.rst_n) in
    let spec = Reg_spec.create ~clock:i.clk ~clear () in
    let%hw accepted = wire 1 in
    let%hw consumed = wire 1 in
    (* Reserve the output register until the host consumes this prediction. *)
    let%hw busy = reg_fb spec ~width:1 ~f:(fun busy -> busy |: accepted &: ~:consumed) in
    let%hw ready = ~:busy &: ~:clear in
    accepted <-- (i.s_tvalid &: ready);
    let prediction =
      Accelerator.hierarchical
        scope
        { clock = i.clk
        ; clear
        ; features = i.s_tdata
        ; features_valid = accepted
        ; weights_write = None
        ; weights_address = None
        ; weights_data = None
        ; weights_commit = None
        }
    in
    let%hw valid =
      reg_fb spec ~width:1 ~f:(fun valid ->
        prediction.prediction_valid |: (valid &: ~:consumed))
    in
    let%hw data = reg spec ~enable:prediction.prediction_valid prediction.prediction in
    consumed <-- (valid &: i.m_tready);
    { O.s_tready = ready; m_tdata = data; m_tvalid = valid &: ~:clear }
  ;;

  let hierarchical ?instance scope i =
    let module H = Hierarchy.In_scope (I) (O) in
    H.hierarchical ?instance ~scope ~name:"fixed_stream" create i
  ;;
end
