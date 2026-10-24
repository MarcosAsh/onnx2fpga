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

  (** Instantiates the unit on the vector [x], returning its output vector. *)
  val connect
    :  Scope.t
    -> Signal.t Clocking.t
    -> Signal.t With_valid.t
    -> Signal.t With_valid.t

  (** The unit alone as a circuit called [name], ports as its own interface names them. *)
  val circuit : Scope.t -> name:string -> Circuit.t
end

(** Raises on a kind with no fixed latency version or a parameter the unit rejects. *)
val of_fixed_node : Design.Node.t -> (module Fixed_instance)
