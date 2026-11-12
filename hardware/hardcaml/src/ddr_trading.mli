open! Core
open! Hardcaml

(** Clears the trading state before replay and drains decisions before completion. *)
module Make (Config : Tick_to_trade.Config) : sig
  module Trading : module type of Tick_to_trade.Make (Config)

  module I : sig
    type 'a t =
      { clock : 'a
      ; clear : 'a
      ; start : 'a
      ; abort_request : 'a
      ; base : 'a
      ; records : 'a
      ; arready : 'a
      ; rvalid : 'a
      ; rdata : 'a
      ; rresp : 'a
      ; rlast : 'a
      }
    [@@deriving hardcaml]
  end

  module O : sig
    type 'a t =
      { replay : 'a Ddr_replay.O.t
      ; trading : 'a Trading.O.t
      ; busy : 'a
      ; done_ : 'a
      ; error : 'a
      ; aborted : 'a
      ; decisions : 'a
      ; orders : 'a
      ; cycles : 'a
      }
    [@@deriving hardcaml]
  end

  val hierarchical : ?instance:string -> Scope.t -> Signal.t I.t -> Signal.t O.t
end

val verilog : Tick_to_trade.Params.t -> string
