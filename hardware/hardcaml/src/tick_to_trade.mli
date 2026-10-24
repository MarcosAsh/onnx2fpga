(** Bytes in to decision out, every decision exactly [latency] cycles after its last byte. *)

open! Core
open! Hardcaml

module Params : sig
  (** [model] is a fixed latency design taking four int8 features and giving one int8. *)
  type t =
    { locates : int list
    ; names : string list
    ; levels : int
    ; order_bits : int
    ; spread_shift : int
    ; flow_shift : int
    ; buy_threshold : int
    ; sell_threshold : int
    ; order_shares : int
    ; model : Design.t
    }
  [@@deriving sexp_of]

  (** The JSON object with these fields, the model's design.json under [model]. *)
  val of_string : string -> t
end

module type Config = sig
  val params : Params.t
end

module Make (_ : Config) : sig
  (** Each unit's cycles, in pipeline order. *)
  val stages : (string * int) list

  val latency : int

  module I : sig
    type 'a t =
      { clock : 'a
      ; clear : 'a
      ; data : 'a
      ; keep : 'a
      ; valid : 'a
      ; last : 'a
      }
    [@@deriving hardcaml]
  end

  (** Each stage's output when valid; [t_out] is the cycle counter. *)
  module O : sig
    type 'a t =
      { ready : 'a
      ; book_valid : 'a
      ; book_symbol : 'a
      ; book_bids : 'a
      ; book_asks : 'a
      ; book_flow : 'a
      ; features_valid : 'a
      ; features : 'a
      ; decision_valid : 'a
      ; action : 'a
      ; prediction : 'a
      ; order_valid : 'a
      ; order : 'a
      ; t_first : 'a
      ; t_last : 'a
      ; t_out : 'a
      }
    [@@deriving hardcaml]
  end

  (** Wait for [ready] after a clear before sending the first beat. *)
  val hierarchical : ?instance:string -> Scope.t -> Signal.t I.t -> Signal.t O.t
end

(** The path with its top named [tick_to_trade], or one unit under a top named [dut]. *)
val verilog : ?unit:string -> Params.t -> string
