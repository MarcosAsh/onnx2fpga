(** Reads up to 16 packed 128-bit records per AXI burst from a 256-byte aligned base.
    Abort drains the outstanding burst before accepting another start. *)

open! Core
open! Hardcaml

module I : sig
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; start : 'a
    ; abort_request : 'a
    ; base : 'a
    ; records : 'a
    ; ready : 'a
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
    { arvalid : 'a
    ; araddr : 'a
    ; arlen : 'a
    ; arsize : 'a
    ; arburst : 'a
    ; rready : 'a
    ; data : 'a
    ; keep : 'a
    ; valid : 'a
    ; last : 'a
    ; busy : 'a
    ; done_ : 'a
    ; error : 'a
    ; aborted : 'a
    ; delivered : 'a
    }
  [@@deriving hardcaml]
end

(** Start requires idle and ready. [clear] must also reset the AXI interconnect, since it
    discards bus state. *)
val hierarchical : ?instance:string -> Scope.t -> Signal.t I.t -> Signal.t O.t
