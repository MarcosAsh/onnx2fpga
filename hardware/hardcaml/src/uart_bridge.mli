(** UART bytes to accelerator beats and back, with byte buffers each way. *)

open! Core
open! Hardcaml

module type Config = sig
  val in_width : int
  val out_width : int
  val rx_depth : int
  val tx_depth : int
end

module Make (_ : Config) : sig
  module Beat_in : sig
    type 'a t =
      { data : 'a
      ; valid : 'a
      }
    [@@deriving hardcaml]
  end

  module Beat_out : sig
    type 'a t =
      { data : 'a
      ; valid : 'a
      }
    [@@deriving hardcaml]
  end

  module Byte : sig
    type 'a t =
      { data : 'a
      ; valid : 'a
      }
    [@@deriving hardcaml]
  end

  module I : sig
    type 'a t =
      { clocking : 'a Clocking.t
      ; rx : 'a With_valid.t
      ; accel_in : 'a Stream.Dest.t
      ; accel_out : 'a Beat_out.t
      ; tx : 'a Stream.Dest.t
      }
    [@@deriving hardcaml]
  end

  module O : sig
    type 'a t =
      { accel_in : 'a Beat_in.t
      ; accel_out : 'a Stream.Dest.t
      ; tx : 'a Byte.t
      ; overflow : 'a
      }
    [@@deriving hardcaml]
  end

  val hierarchical : ?instance:string -> Scope.t -> Signal.t I.t -> Signal.t O.t
end
