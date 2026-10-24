(** Fully unrolled dense layer with fused bias and rescale, one vector per cycle. *)

open! Core
open! Hardcaml

module type Config = sig
  (** Input elements. *)
  val mw : int

  (** Output channels. *)
  val mh : int

  val act_bits : int
  val wgt_bits : int
  val acc_bits : int
  val out_bits : int
  val mult_bits : int
  val shift : int
  val out_min : int
  val out_max : int
  val out_zp : int

  (** [`Constant] builds the products from shifts and adds; values do not change. *)
  val arithmetic : [ `Multiplier | `Constant ]

  (** One word per output channel, element k of channel c's word being w[k][c]. *)
  val weights : Bits.t list

  (** One [acc_bits] word per output channel. *)
  val bias : Bits.t list

  (** One [mult_bits] word per output channel. *)
  val mult : Bits.t list
end

module Make (_ : Config) : sig
  (** Cycles from [x.valid] to [y.valid]: products, then the sum, then the rescale. *)
  val latency : int

  module I : sig
    type 'a t =
      { clocking : 'a Clocking.t
      ; x : 'a With_valid.t
      }
    [@@deriving hardcaml]
  end

  module O : sig
    type 'a t = { y : 'a With_valid.t } [@@deriving hardcaml]
  end

  (** [y] is [x] through the layer [latency] cycles later; [clear] only clears valid. *)
  val hierarchical : ?instance:string -> Scope.t -> Signal.t I.t -> Signal.t O.t
end
