(** A fixed model with one inference in flight and a held output under backpressure. *)

open! Core
open! Hardcaml

module type Config = sig
  val design : Design.t
end

module Make (_ : Config) : sig
  module I : sig
    type 'a t =
      { clk : 'a
      ; rst_n : 'a
      ; s_tdata : 'a
      ; s_tvalid : 'a
      ; m_tready : 'a
      }
    [@@deriving hardcaml]
  end

  module O : sig
    type 'a t =
      { s_tready : 'a
      ; m_tdata : 'a
      ; m_tvalid : 'a
      }
    [@@deriving hardcaml]
  end

  (** Reset drops pending input and output. Reloadable weights are rejected. *)
  val hierarchical : ?instance:string -> Scope.t -> Signal.t I.t -> Signal.t O.t
end
