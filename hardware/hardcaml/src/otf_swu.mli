(** Frame buffered sliding window generator, channel last. *)

open! Core
open! Hardcaml

module type Config = sig
  val ifm_h : int
  val ifm_w : int
  val ifm_ch : int
  val ofm_h : int
  val ofm_w : int
  val kh : int
  val kw : int
  val stride_h : int
  val stride_w : int
  val pad_t : int
  val pad_l : int
  val dil_h : int
  val dil_w : int
  val simd : int
  val data_bits : int
  val pad_value : int
end

module Make (_ : Config) : sig
  module Beat : sig
    type 'a t =
      { data : 'a
      ; valid : 'a
      }
    [@@deriving hardcaml]
  end

  module I : sig
    type 'a t =
      { clocking : 'a Clocking.t
      ; s : 'a Beat.t
      ; m : 'a Stream.Dest.t
      }
    [@@deriving hardcaml]
  end

  module O : sig
    type 'a t =
      { s : 'a Stream.Dest.t
      ; m : 'a Beat.t
      }
    [@@deriving hardcaml]
  end

  val hierarchical : ?instance:string -> Scope.t -> Signal.t I.t -> Signal.t O.t
end
