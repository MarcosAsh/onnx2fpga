open! Core
open! Hardcaml
open! Onnx2fpga_hardcaml

let design_command =
  Command.basic
    ~summary:"Verilog for a compiled design, from the design.json onnx2fpga writes"
    [%map_open.Command
      let path = anon ("DESIGN_JSON" %: string) in
      fun () ->
        let design = Design.of_string (In_channel.read_all path) in
        match design.mode with
        | Streaming -> print_string (Top.verilog design)
        | Fixed _ -> print_string (Fixed_top.verilog design)]
;;

let units_command =
  Command.basic
    ~summary:
      "Verilog for units on their own, each a top level named by its instance field, \
       from a design.json whose nodes list them"
    [%map_open.Command
      let path = anon ("DESIGN_JSON" %: string) in
      fun () ->
        let design = Design.of_string (In_channel.read_all path) in
        let scope = Scope.create ~auto_label_hierarchical_ports:true () in
        let circuits =
          List.map design.nodes ~f:(fun node ->
            match design.mode with
            | Streaming ->
              let (module Unit) = Units.of_node node in
              Unit.circuit scope ~name:node.instance
            | Fixed _ ->
              let (module Unit) = Units.of_fixed_node node in
              Unit.circuit scope ~name:node.instance)
        in
        Rtl.create ~database:(Scope.circuit_database scope) Verilog circuits
        |> Rtl.full_hierarchy
        |> Rope.to_string
        |> print_string]
;;

let icepi_command =
  Command.basic
    ~summary:"Verilog for a compiled design on an Icepi Zero, behind its USB serial port"
    [%map_open.Command
      let path = anon ("DESIGN_JSON" %: string)
      and mhz =
        flag
          "mhz"
          (optional_with_default 20 int)
          ~doc:"MHZ the PLL clocks the design at, a divisor of 480 (default 20)"
      and baud =
        flag "baud" (optional_with_default 1_000_000 int) ~doc:"BAUD (default 1000000)"
      and rx_depth =
        flag
          "rx-depth"
          (optional_with_default 2048 int)
          ~doc:"BYTES of input buffer (default 2048)"
      and tx_depth =
        flag
          "tx-depth"
          (optional_with_default 2048 int)
          ~doc:"BYTES of output buffer (default 2048)"
      in
      fun () ->
        let design = Design.of_string (In_channel.read_all path) in
        print_string (Icepi_top.verilog ~design ~mhz ~baud ~rx_depth ~tx_depth)]
;;

let tick_to_trade_command =
  Command.basic
    ~summary:"Verilog for the tick to trade path, from a JSON config holding the model"
    [%map_open.Command
      let path = anon ("CONFIG_JSON" %: string)
      and unit =
        flag
          "unit"
          (optional string)
          ~doc:"NAME only this unit, moldudp64_step or book_side, under a top named dut"
      and stages = flag "stages" no_arg ~doc:" print each unit's latency instead" in
      fun () ->
        let params = Tick_to_trade.Params.of_string (In_channel.read_all path) in
        if stages
        then
          let module T =
            Tick_to_trade.Make (struct
              let params = params
            end)
          in
          print_s [%sexp (T.stages : (string * int) list)]
        else print_string (Tick_to_trade.verilog ?unit params)]
;;

let () =
  Command_unix.run
    (Command.group
       ~summary:""
       [ "design", design_command
       ; "units", units_command
       ; "icepi", icepi_command
       ; "tick-to-trade", tick_to_trade_command
       ])
;;
