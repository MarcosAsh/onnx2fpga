open! Core
open! Hardcaml
open! Signal

module Make (Config : Tick_to_trade.Config) = struct
  module Pipeline = Ddr_trading.Make (Config)

  module I = struct
    type 'a t =
      { clock : 'a
      ; clear : 'a
      ; s_axi_awaddr : 'a [@bits 16]
      ; s_axi_awvalid : 'a
      ; s_axi_wdata : 'a [@bits 32]
      ; s_axi_wstrb : 'a [@bits 4]
      ; s_axi_wvalid : 'a
      ; s_axi_bready : 'a
      ; s_axi_araddr : 'a [@bits 16]
      ; s_axi_arvalid : 'a
      ; s_axi_rready : 'a
      ; arready : 'a
      ; rdata : 'a [@bits 128]
      ; rvalid : 'a
      ; rresp : 'a [@bits 2]
      ; rlast : 'a
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t =
      { s_axi_awready : 'a
      ; s_axi_wready : 'a
      ; s_axi_bvalid : 'a
      ; s_axi_bresp : 'a [@bits 2]
      ; s_axi_arready : 'a
      ; s_axi_rvalid : 'a
      ; s_axi_rdata : 'a [@bits 32]
      ; s_axi_rresp : 'a [@bits 2]
      ; pipeline : 'a Pipeline.O.t
      }
    [@@deriving hardcaml]
  end

  let create scope (i : Signal.t I.t) =
    let spec = Reg_spec.create ~clock:i.clock ~clear:i.clear () in
    let%hw write = wire 1 in
    let%hw bvalid =
      reg_fb spec ~width:1 ~f:(fun v -> write |: (v &: ~:(i.s_axi_bready)))
    in
    let%hw aw_take = wire 1 in
    let%hw w_take = wire 1 in
    let%hw aw_full = reg_fb spec ~width:1 ~f:(fun v -> v |: aw_take &: ~:write) in
    let%hw w_full = reg_fb spec ~width:1 ~f:(fun v -> v |: w_take &: ~:write) in
    let%hw awready = ~:aw_full &: ~:bvalid &: ~:(i.clear) in
    let%hw wready = ~:w_full &: ~:bvalid &: ~:(i.clear) in
    aw_take <-- (awready &: i.s_axi_awvalid);
    w_take <-- (wready &: i.s_axi_wvalid);
    let%hw address = reg spec ~enable:aw_take i.s_axi_awaddr in
    let%hw data = reg spec ~enable:w_take i.s_axi_wdata in
    let%hw strobe = reg spec ~enable:w_take i.s_axi_wstrb in
    write <-- (aw_full &: w_full &: ~:bvalid &: ~:(i.clear));
    let writes offset = write &: (address ==:. offset) in
    let register offset =
      List.init 4 ~f:(fun byte ->
        reg
          spec
          ~enable:(writes offset &: strobe.:(byte))
          data.:[(byte * 8) + 7, byte * 8])
      |> concat_lsb
    in
    let%hw base_low = register 0x20 in
    let%hw base_high = register 0x24 in
    let%hw records = register 0x28 in
    let%hw control = writes 0x10 &: strobe.:(0) in
    let%hw busy = wire 1 in
    let%hw start_denied = busy |: (base_high.:[31, 8] <>:. 0) in
    let%hw start = reg spec (control &: data.:(0) &: ~:(data.:(1)) &: ~:start_denied) in
    let%hw abort_request = reg spec (control &: data.:(1)) in
    let pipeline =
      Pipeline.hierarchical
        scope
        { clock = i.clock
        ; clear = i.clear
        ; start
        ; abort_request
        ; base = base_high.:[7, 0] @: base_low
        ; records
        ; arready = i.arready
        ; rdata = i.rdata
        ; rvalid = i.rvalid
        ; rresp = i.rresp
        ; rlast = i.rlast
        }
    in
    busy <-- pipeline.busy;
    let%hw known_write =
      address
      ==:. 0x10
      |: (address ==:. 0x20)
      |: (address ==:. 0x24)
      |: (address ==:. 0x28)
    in
    let%hw denied =
      ~:known_write |: (control &: data.:(0) &: ~:(data.:(1)) &: start_denied)
    in
    let%hw bresp =
      reg spec ~enable:write (mux2 denied (of_int_trunc ~width:2 2) (zero 2))
    in
    let%hw rvalid = wire 1 in
    let%hw arready = ~:rvalid &: ~:(i.clear) in
    let%hw read = arready &: i.s_axi_arvalid in
    rvalid <-- reg spec (read |: (rvalid &: ~:(i.s_axi_rready)));
    let status =
      concat_lsb
        [ pipeline.busy; pipeline.done_; pipeline.error; pipeline.aborted; zero 28 ]
    in
    let snapshot_spec = Reg_spec.create ~clock:i.clock ~clear:(i.clear |: start) () in
    let snapshot value =
      reg snapshot_spec ~enable:pipeline.trading.decision_valid value
    in
    let%hw result =
      snapshot
        (concat_lsb
           [ pipeline.trading.prediction
           ; pipeline.trading.action
           ; pipeline.trading.order_valid
           ; zero 21
           ])
    in
    let%hw order =
      snapshot (mux2 pipeline.trading.order_valid pipeline.trading.order (zero 376))
    in
    let order_words = split_lsb ~part_width:32 (uresize order ~width:384) in
    let values =
      [ 0x00, of_int_trunc ~width:32 0x4f544432
      ; 0x04, of_int_trunc ~width:32 1
      ; 0x14, status
      ; 0x20, base_low
      ; 0x24, base_high
      ; 0x28, records
      ; 0x30, pipeline.replay.delivered
      ; 0x34, pipeline.decisions
      ; 0x38, pipeline.orders
      ; 0x3c, pipeline.cycles
      ; 0x40, of_int_trunc ~width:32 Pipeline.Trading.latency
      ; 0x44, result
      ; 0x48, snapshot pipeline.trading.t_first
      ; 0x4c, snapshot pipeline.trading.t_last
      ; 0x50, snapshot pipeline.trading.t_out
      ]
      @ List.mapi order_words ~f:(fun index value -> 0x100 + (index * 4), value)
    in
    let%hw known_read =
      reduce
        ~f:( |: )
        (List.map values ~f:(fun (offset, _) -> i.s_axi_araddr ==:. offset))
    in
    let%hw selected =
      cases
        ~default:(zero 32)
        i.s_axi_araddr
        (List.map values ~f:(fun (offset, value) -> of_int_trunc ~width:16 offset, value))
    in
    { O.s_axi_awready = awready
    ; s_axi_wready = wready
    ; s_axi_bvalid = bvalid
    ; s_axi_bresp = bresp
    ; s_axi_arready = arready
    ; s_axi_rvalid = rvalid
    ; s_axi_rdata = reg spec ~enable:read selected
    ; s_axi_rresp =
        reg spec ~enable:read (mux2 known_read (zero 2) (of_int_trunc ~width:2 3))
    ; pipeline
    }
  ;;

  let hierarchical ?instance scope i =
    let module H = Hierarchy.In_scope (I) (O) in
    H.hierarchical ?instance ~scope ~name:"ddr_regs" create i
  ;;
end

let verilog params =
  let module T =
    Make (struct
      let params = params
    end)
  in
  let module C = Circuit.With_interface (T.I) (T.O) in
  let scope = Scope.create ~auto_label_hierarchical_ports:true () in
  let circuit = C.create_exn ~name:"ddr_regs_sim" (T.hierarchical scope) in
  Rtl.create ~database:(Scope.circuit_database scope) Verilog [ circuit ]
  |> Rtl.full_hierarchy
  |> Rope.to_string
;;
