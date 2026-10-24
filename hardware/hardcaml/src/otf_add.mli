(** Combinational elementwise add of two streams scaled onto the output grid. *)

open! Core
open! Hardcaml

module type Config = sig
  val pe : int
  val in_bits : int
  val out_bits : int
  val acc_bits : int
  val left_mult : int
  val right_mult : int
  val shift : int
  val out_min : int
  val out_max : int
  val bias : int
  val out_zp : int
end

module Make (_ : Config) : sig
  module In : sig
    type 'a t =
      { data : 'a
      ; valid : 'a
      }
    [@@deriving hardcaml]
  end

  module Out : sig
    type 'a t =
      { data : 'a
      ; valid : 'a
      }
    [@@deriving hardcaml]
  end

  module I : sig
    type 'a t =
      { clocking : 'a Clocking.t
      ; a : 'a In.t
      ; b : 'a In.t
      ; m : 'a Stream.Dest.t
      }
    [@@deriving hardcaml]
  end

  module O : sig
    type 'a t =
      { a : 'a Stream.Dest.t
      ; b : 'a Stream.Dest.t
      ; m : 'a Out.t
      }
    [@@deriving hardcaml]
  end

  val hierarchical : ?instance:string -> Scope.t -> Signal.t I.t -> Signal.t O.t
end
