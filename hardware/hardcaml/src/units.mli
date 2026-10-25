(** Elaborates a unit of a {!Design.t} from its runtime parameters. *)

open! Core
open! Hardcaml

(** One stream as wires: the producer drives [data] and [valid], the consumer [ready]. *)
module Port : sig
  type t =
    { data : Signal.t
    ; valid : Signal.t
    ; ready : Signal.t
    }
end

module type Instance = sig
  (** Instantiates the unit; ports follow the node's port order. *)
  val connect
    :  Scope.t
    -> Signal.t Clocking.t
    -> inputs:Port.t list
    -> outputs:Port.t list
    -> unit

  (** The unit alone as a circuit called [name], ports as its own interface names them. *)
  val circuit : Scope.t -> name:string -> Circuit.t
end

(** Raises on an unknown kind or a parameter the unit rejects. *)
val of_node : Design.Node.t -> (module Instance)

(** A fixed latency unit: one vector in and one out on every cycle, with no ready. *)
module type Fixed_instance = sig
  (** Cycles from a valid input to its valid output. *)
  val latency : int

  (** Words the unit keeps in each weight bank, 0 when its weights are frozen. *)
  val words : int

  (** Its compiled bank words, empty when frozen. *)
  val image : Bits.t list

  (** Instantiates the unit on its inputs, in step, and the bank tag frozen units only
      delay. *)
  val connect
    :  Scope.t
    -> Signal.t Clocking.t
    -> store:Signal.t Weight_banks.Write.t option
    -> bank:Signal.t option
    -> Signal.t With_valid.t list
    -> Signal.t With_valid.t * Signal.t option

  (** The unit alone as a circuit called [name], ports as its own interface names them. *)
  val circuit : Scope.t -> name:string -> Circuit.t
end

(** Raises on a kind with no fixed version; [base] is a reloadable unit's first bank word. *)
val of_fixed_node : ?base:int -> Design.Node.t -> (module Fixed_instance)
