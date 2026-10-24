(** 8N1 receiver sampling mid bit; [line_break] is the line low for twenty bit times. *)

open! Core
open! Hardcaml

module type Config = sig
  (** At least 4; tolerates about 4 percent baud error at 16 clocks a bit. *)
  val clocks_per_bit : int
end

module Make (_ : Config) : sig
  module I : sig
    type 'a t =
      { clocking : 'a Clocking.t
      ; rxd : 'a
      }
    [@@deriving hardcaml]
  end

  module O : sig
    type 'a t =
      { byte : 'a With_valid.t
      ; framing_error : 'a
      ; line_break : 'a
      }
    [@@deriving hardcaml]
  end

  val hierarchical : ?instance:string -> Scope.t -> Signal.t I.t -> Signal.t O.t
end
