(** The whole accelerator: a {!Design.t} wired stream to stream; [rst_n] is active low. *)

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

  val hierarchical : ?instance:string -> Scope.t -> Signal.t I.t -> Signal.t O.t
end

(** One Verilog file holding the top level, named [design.top], and every module under it. *)
val verilog : Design.t -> string
