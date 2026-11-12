open! Core
open! Hardcaml
open! Signal

module type Config = sig
  val address_bits : int
  val data_bits : int
  val use_ultra : bool
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
    let%hw address =
      mux2
        initialize
        (zero Config.address_bits)
        (mux2 prefer_b write_b.address write_a.address)
    in
    let%hw data = mux2 prefer_b write_b.data write_a.data in
    let%hw create_a = write_a.enable &: lsb write_a.data in
    let%hw create_b = write_b.enable &: lsb write_b.data in
    let%hw creating = create_a |: create_b in
    let%hw clearing = write_a.enable &: ~:(lsb write_a.data) in
    let bank_bits = Int.min 8 ((Config.address_bits + 1) / 2) in
    let select_bits = Config.address_bits - bank_bits in
    let bank_count = 1 lsl select_bits in
    let low address = sel_bottom address ~width:bank_bits in
    let bank address =
      if select_bits = 0 then gnd else sel_top address ~width:select_bits
    in
    let in_bank address index =
      if select_bits = 0 then vdd else bank address ==:. index
    in
    let created = Array.init bank_count ~f:(fun _ -> Array.init 4 ~f:(fun _ -> wire 1)) in
    let deleted = Array.init bank_count ~f:(fun _ -> Array.init 4 ~f:(fun _ -> wire 1)) in
    let read_addresses = [| read_a; read_b; write_a.address; write_b.address |] in
    let create_address = mux2 initialize write_a.address address in
    let%hw pending_created_address = reg spec create_address in
    let%hw pending_deleted_address = reg spec write_a.address in
    let packed f = concat_lsb (List.init bank_count ~f) in
    let%hw pending_created_enable =
      packed (fun bank ->
        reg spec (initialize |: creating &: in_bank create_address bank))
    in
    let%hw pending_deleted_enable =
      packed (fun bank ->
        reg spec (initialize |: clearing &: in_bank write_a.address bank))
    in
    let%hw pending_created_data =
      packed (fun bank ->
        reg
          spec
          (mux2 initialize gnd ~:(mux2 prefer_b deleted.(bank).(3) deleted.(bank).(2))))
    in
    let%hw pending_deleted_data =
      packed (fun bank -> reg spec (mux2 initialize gnd created.(bank).(2)))
    in
    let flags name ~enable ~address ~data =
      multiport_memory
        ~name
        ~attributes:
          [ Rtl_attribute.create
              "ram_style"
              ~applies_to:[ Memories ]
              ~value:(String "distributed")
          ]
        (1 lsl bank_bits)
        ~write_ports:
          [| { Write_port.write_clock = clock
             ; write_enable = enable
             ; write_address = low address
             ; write_data = data
             }
          |]
        ~read_addresses:(Array.map read_addresses ~f:low)
    in
    Array.iteri created ~f:(fun bank _ ->
      let created_enable = pending_created_enable.:(bank) in
      let deleted_enable = pending_deleted_enable.:(bank) in
      let created_data = pending_created_data.:(bank) in
      let deleted_data = pending_deleted_data.:(bank) in
      let next_created =
        flags
          ("created_" ^ Int.to_string bank)
          ~enable:created_enable
          ~address:pending_created_address
          ~data:created_data
      in
      let next_deleted =
        flags
          ("deleted_" ^ Int.to_string bank)
          ~enable:deleted_enable
          ~address:pending_deleted_address
          ~data:deleted_data
      in
      Array.iteri read_addresses ~f:(fun index address ->
        created.(bank).(index)
        <-- mux2
              (created_enable &: (low address ==: low pending_created_address))
              created_data
              next_created.(index);
        deleted.(bank).(index)
        <-- mux2
              (deleted_enable &: (low address ==: low pending_deleted_address))
              deleted_data
              next_deleted.(index)));
    let payload =
      if Config.use_ultra
      then
        Array.mapi [| read_a; read_b |] ~f:(fun port read_address ->
          split_lsb ~exact:false ~part_width:72 (msbs data)
          |> List.mapi ~f:(fun bank write_data ->
            let memory =
              multiport_memory
                ~name:("payload_" ^ Int.to_string port ^ "_" ^ Int.to_string bank)
                ~attributes:
                  [ Rtl_attribute.create
                      "ram_style"
                      ~applies_to:[ Memories ]
                      ~value:(String "ultra")
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
                ~read_addresses:[| read_address |]
            in
            memory.(0))
          |> concat_lsb)
      else (
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
        Array.init 2 ~f:(fun port ->
          concat_lsb (List.map payload_banks ~f:(fun bank -> bank.(port)))))
    in
    let forwarded = reg spec (msbs data) in
    let live_after_write read_address previous =
      let a = read_address ==: write_a.address in
      let b = read_address ==: write_b.address in
      mux2
        (initialize &: a)
        gnd
        (mux2
           (~:initialize &: write_a.enable &: a)
           (lsb write_a.data)
           (mux2 (~:initialize &: write_b.enable &: b) (lsb write_b.data) previous))
    in
    let read index read_address =
      let live_banks =
        List.init bank_count ~f:(fun bank ->
          reg spec (created.(bank).(index) ^: deleted.(bank).(index)))
      in
      let live =
        if bank_count = 1
        then List.hd_exn live_banks
        else mux (reg spec (bank read_address)) live_banks
      in
      let override_a =
        reg spec (initialize |: write_a.enable &: (read_address ==: write_a.address))
      in
      let override_b =
        reg spec (~:initialize &: write_b.enable &: (read_address ==: write_b.address))
      in
      let value_a = reg spec (mux2 initialize gnd (lsb write_a.data)) in
      let value_b = reg spec (lsb write_b.data) in
      let live = mux2 override_a value_a (mux2 override_b value_b live) in
      let bypass =
        reg spec (payload_write &: ~:initialize &: (read_address ==: address))
      in
      let data =
        mux2 (reg spec bypass) (reg spec forwarded) (reg spec (reg spec payload.(index)))
      in
      let delayed_address = reg spec read_address in
      let bypass =
        reg spec (payload_write &: ~:initialize &: (delayed_address ==: address))
      in
      mux2 bypass forwarded data @: reg spec (live_after_write delayed_address live)
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
