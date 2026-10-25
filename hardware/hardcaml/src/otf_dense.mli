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

module type Reloadable_config = sig
  include Config

  (** Word address of this layer's first value in the banks. *)
  val base : int
end

(** The same layer with its values in two banks a host rewrites; needs [`Multiplier]. *)
module Make_reloadable (_ : Reloadable_config) : sig
  (** Three cycles, the same as frozen. *)
  val latency : int

  (** Values in a bank: weights channel by channel, then the biases, then the multipliers. *)
  val words : int

  (** The compiled values, which both banks start with, one 32 bit word each. *)
  val image : Bits.t list

  module I : sig
    type 'a t =
      { clocking : 'a Clocking.t
      ; x : 'a With_valid.t
      ; x_bank : 'a
      ; store : 'a Weight_banks.Write.t
      }
    [@@deriving hardcaml]
  end

  module O : sig
    type 'a t =
      { y : 'a With_valid.t
      ; y_bank : 'a
      }
    [@@deriving hardcaml]
  end

  (** Each stage uses the bank [x_bank] named when [x] arrived, which leaves as [y_bank]. *)
  val hierarchical : ?instance:string -> Scope.t -> Signal.t I.t -> Signal.t O.t
end
