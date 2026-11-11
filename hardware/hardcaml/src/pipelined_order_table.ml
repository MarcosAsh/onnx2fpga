open! Core
open! Hardcaml
open! Signal

module type Config = sig
  val address_bits : int
  val data_bits : int
end

module Make (Config : Config) = struct
  module Write = struct
    type 'a t =
      { address : 'a [@bits Config.address_bits]
      ; data : 'a [@bits Config.data_bits]
      ; enable : 'a
      }
    [@@deriving hardcaml]
  end

  module I = struct
    type 'a t =
      { clock : 'a
      ; initialize : 'a
      ; prefer_b : 'a
      ; payload_write : 'a
      ; write_a : 'a Write.t
      ; write_b : 'a Write.t
      ; read_a : 'a [@bits Config.address_bits]
      ; read_b : 'a [@bits Config.address_bits]
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t =
      { qa : 'a [@bits Config.data_bits]
      ; qb : 'a [@bits Config.data_bits]
      ; qa_first : 'a [@bits Config.data_bits]
      ; qb_first : 'a [@bits Config.data_bits]
      }
    [@@deriving hardcaml]
  end

  let create
    scope
    ({ clock; initialize; prefer_b; payload_write; write_a; write_b; read_a; read_b } :
      Signal.t I.t)
    =
    let spec = Reg_spec.create ~clock () in
    let%hw address = mux2 prefer_b write_b.address write_a.address in
    let%hw data = mux2 prefer_b write_b.data write_a.data in
    let%hw create_a = write_a.enable &: lsb write_a.data in
    let%hw create_b = write_b.enable &: lsb write_b.data in
    let%hw creating = create_a |: create_b in
    let%hw clearing = write_a.enable &: ~:(lsb write_a.data) in
    let created = Array.init 4 ~f:(fun _ -> wire 1) in
    let deleted = Array.init 4 ~f:(fun _ -> wire 1) in
    let read_addresses = [| read_a; read_b; write_a.address; write_b.address |] in
    let%hw pending_created_enable = reg spec (initialize |: creating) in
    let%hw pending_created_address = reg spec (mux2 initialize write_a.address address) in
    let%hw pending_created_data =
      reg spec (mux2 initialize gnd ~:(mux2 prefer_b deleted.(3) deleted.(2)))
    in
    let%hw pending_deleted_enable = reg spec (initialize |: clearing) in
    let%hw pending_deleted_address = reg spec write_a.address in
    let%hw pending_deleted_data = reg spec (mux2 initialize gnd created.(2)) in
    let flags name ~enable ~address ~data =
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
    let next_created =
      flags
        "created"
        ~enable:pending_created_enable
        ~address:pending_created_address
        ~data:pending_created_data
    in
    let next_deleted =
      flags
        "deleted"
        ~enable:pending_deleted_enable
        ~address:pending_deleted_address
        ~data:pending_deleted_data
    in
    Array.iteri read_addresses ~f:(fun index address ->
      created.(index)
      <-- mux2
            (pending_created_enable &: (address ==: pending_created_address))
            pending_created_data
            next_created.(index);
      deleted.(index)
      <-- mux2
            (pending_deleted_enable &: (address ==: pending_deleted_address))
            pending_deleted_data
            next_deleted.(index));
    let payload_banks =
      split_lsb ~exact:false ~part_width:9 (msbs data)
      |> List.mapi ~f:(fun bank write_data ->
        multiport_memory
          ~name:("payload_" ^ Int.to_string bank)
          ~attributes:
            [ Rtl_attribute.Vivado.Ram_style.block
            ; Rtl_attribute.Vivado.cascade_height 1
            ]
          (1 lsl Config.address_bits)
          ~write_ports:
            [| { Write_port.write_clock = clock
               ; write_enable = payload_write &: ~:initialize
               ; write_address = address
               ; write_data
               }
            |]
          ~read_addresses:[| read_a; read_b |])
    in
    let payload =
      Array.init 2 ~f:(fun port ->
        concat_lsb (List.map payload_banks ~f:(fun bank -> bank.(port))))
    in
    let forwarded = reg spec (msbs data) in
    let read index read_address =
      let a = read_address ==: write_a.address in
      let b = read_address ==: write_b.address in
      let live =
        mux2
          (initialize &: a)
          gnd
          (mux2
             (~:initialize &: write_a.enable &: a)
             (lsb write_a.data)
             (mux2
                (~:initialize &: write_b.enable &: b)
                (lsb write_b.data)
                (created.(index) ^: deleted.(index))))
      in
      let bypass =
        reg spec (payload_write &: ~:initialize &: (read_address ==: address))
      in
      mux2 bypass forwarded (reg spec payload.(index)) @: reg spec live
    in
    let qa_first = read 0 read_a in
    let qb_first = read 1 read_b in
    { O.qa = reg spec qa_first; qb = reg spec qb_first; qa_first; qb_first }
  ;;

  let hierarchical ?instance scope i =
    let module H = Hierarchy.In_scope (I) (O) in
    H.hierarchical ?instance ~scope ~name:"pipelined_order_table" create i
  ;;
end
