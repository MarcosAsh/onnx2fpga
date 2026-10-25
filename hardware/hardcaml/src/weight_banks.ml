open! Core
open! Hardcaml
open! Signal

let address_bits = 13

module Write = struct
  type 'a t =
    { write : 'a
    ; bank : 'a
    ; address : 'a [@bits address_bits]
    ; data : 'a [@bits 32]
    }
  [@@deriving hardcaml]
end

(* zlib's CRC-32: reflected, polynomial 0xEDB88320, the word's bit 0 first. *)
let crc_step (type a) (module C : Comb.S with type t = a) (state : a) (data : a) =
  List.fold (C.bits_lsb data) ~init:state ~f:(fun state data_bit ->
    let feedback = C.repeat C.(lsb state ^: data_bit) ~count:32 in
    C.(srl state ~by:1 ^: (feedback &: of_unsigned_int ~width:32 0xedb88320)))
;;

let crc_state words = List.fold words ~init:(Bits.ones 32) ~f:(crc_step (module Bits))
let crc words = Bits.to_unsigned_int Bits.(~:(crc_state words))

module type Config = sig
  val image : Bits.t list
end

module Make (Config : Config) = struct
  open Config

  let () =
    if List.length image > 1 lsl address_bits
    then
      raise_s
        [%message
          "the reloadable weights do not fit the register window"
            ~words:(List.length image : int)]
  ;;

  module I = struct
    type 'a t =
      { clock : 'a
      ; write : 'a
      ; address : 'a [@bits address_bits]
      ; data : 'a [@bits 32]
      ; commit : 'a
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t =
      { bank : 'a
      ; version : 'a [@bits 32]
      ; crc : 'a [@bits 32]
      ; idle_crc : 'a [@bits 32]
      }
    [@@deriving hardcaml]
  end

  let create (scope : Scope.t) (i : Signal.t I.t) =
    let spec = Reg_spec.create ~clock:i.clock () in
    let%hw bank =
      reg_fb spec ~initialize_to:Bits.gnd ~width:1 ~f:(fun bank -> bank ^: i.commit)
    in
    let%hw version =
      reg_fb spec ~initialize_to:(Bits.zero 32) ~enable:i.commit ~width:32 ~f:(fun v ->
        v +:. 1)
    in
    (* A write to word 0 starts a bank's CRC afresh, so a host can always start over. *)
    let%hw restart = i.address ==:. 0 in
    let bank_crc index =
      reg_fb
        spec
        ~initialize_to:(crc_state image)
        ~enable:(i.write &: (bank ==:. 1 - index))
        ~width:32
        ~f:(fun state -> crc_step (module Signal) (mux2 restart (ones 32) state) i.data)
    in
    let%hw crc0 = bank_crc 0 in
    let%hw crc1 = bank_crc 1 in
    { O.bank; version; crc = ~:(mux2 bank crc1 crc0); idle_crc = ~:(mux2 bank crc0 crc1) }
  ;;

  let hierarchical ?instance scope i =
    let module H = Hierarchy.In_scope (I) (O) in
    H.hierarchical ?instance ~scope ~name:"weight_banks" create i
  ;;
end
