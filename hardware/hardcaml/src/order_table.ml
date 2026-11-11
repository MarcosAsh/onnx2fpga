open! Core
open! Hardcaml
open! Signal

module type Config = sig
  val address_bits : int
  val data_bits : int
end

module Make (Config : Config) = struct
  module Port = struct
    type 'a t =
      { address : 'a [@bits Config.address_bits]
      ; data : 'a [@bits Config.data_bits]
      ; enable : 'a
      ; write : 'a
      }
    [@@deriving hardcaml]
  end

  module I = struct
    type 'a t =
      { clock : 'a
      ; initialize : 'a
      ; prefer_b : 'a
      ; payload_write : 'a
      ; port_a : 'a Port.t
      ; port_b : 'a Port.t
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t =
      { qa : 'a [@bits Config.data_bits]
      ; qb : 'a [@bits Config.data_bits]
      }
    [@@deriving hardcaml]
  end

  let create
    scope
    ({ clock; initialize; prefer_b; payload_write; port_a; port_b } : Signal.t I.t)
    =
    let spec = Reg_spec.create ~clock () in
    let read_addresses = [| port_a.address; port_b.address |] in
    let created = Array.init 2 ~f:(fun _ -> wire 1) in
    let deleted = Array.init 2 ~f:(fun _ -> wire 1) in
    let memory name ~enable ~address ~data =
      multiport_memory
        ~name
        ~attributes:
          [ Rtl_attribute.create
              "ram_style"
              ~applies_to:[ Memories ]
              ~value:(String "distributed")
          ]
        (1 lsl Config.address_bits)
        ~write_ports:
          [| { Write_port.write_clock = clock
             ; write_enable = enable
             ; write_address = address
             ; write_data = data
             }
          |]
        ~read_addresses
    in
    let%hw write_a = port_a.enable &: port_a.write in
    let%hw write_b = port_b.enable &: port_b.write in
    let%hw create_a = write_a &: lsb port_a.data in
    let%hw create_b = write_b &: lsb port_b.data in
    let%hw clearing = write_a &: ~:(lsb port_a.data) in
    let%hw creating = create_a |: create_b in
    let%hw write_address = mux2 prefer_b port_b.address port_a.address in
    let payload =
      memory
        "payload"
        ~enable:(payload_write &: ~:initialize)
        ~address:write_address
        ~data:(msbs (mux2 prefer_b port_b.data port_a.data))
    in
    let next_created =
      memory
        "created"
        ~enable:(initialize |: creating)
        ~address:(mux2 initialize port_a.address write_address)
        ~data:(mux2 initialize gnd ~:(mux2 prefer_b deleted.(1) deleted.(0)))
    in
    let next_deleted =
      memory
        "deleted"
        ~enable:(initialize |: clearing)
        ~address:port_a.address
        ~data:(mux2 initialize gnd created.(0))
    in
    Array.iter2_exn created next_created ~f:( <-- );
    Array.iter2_exn deleted next_deleted ~f:( <-- );
    let read index (port : _ Port.t) =
      reg
        spec
        ~enable:(port.enable &: ~:(port.write))
        (payload.(index) @: created.(index) ^: deleted.(index))
    in
    { O.qa = read 0 port_a; qb = read 1 port_b }
  ;;

  let hierarchical ?instance scope i =
    let module H = Hierarchy.In_scope (I) (O) in
    H.hierarchical ?instance ~scope ~name:"order_table" create i
  ;;
end
