open! Core
open! Hardcaml

module Dest = struct
  type 'a t = { ready : 'a } [@@deriving hardcaml]
end

module type Width = sig
  val width : int
end

module Source (Width : Width) = struct
  type 'a t =
    { data : 'a [@bits Width.width]
    ; valid : 'a
    }
  [@@deriving hardcaml]
end
