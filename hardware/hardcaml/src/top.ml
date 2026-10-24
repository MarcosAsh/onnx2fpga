open! Core
open! Hardcaml
open! Signal

module type Config = sig
  val design : Design.t
end

module Make (Config : Config) = struct
  open Config

  module I = struct
    type 'a t =
      { clk : 'a
      ; rst_n : 'a
      ; s_tdata : 'a [@bits Design.width design design.input]
      ; s_tvalid : 'a
      ; m_tready : 'a
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t =
      { s_tready : 'a
      ; m_tdata : 'a [@bits Design.width design design.output]
      ; m_tvalid : 'a
      }
    [@@deriving hardcaml]
  end

  let identifier name =
    String.map name ~f:(fun c -> if Char.is_alphanum c then c else '_')
  ;;

  let create (scope : Scope.t) (i : Signal.t I.t) =
    let%hw clear = ~:(i.rst_n) in
    let clocking = { Clocking.clock = i.clk; clear } in
    (* Internal streams keep the tensor names the SystemVerilog netlist gives them. *)
    let ports =
      List.map design.streams ~f:(fun { name; width } ->
        let net suffix width =
          Scope.naming scope (wire width) (identifier name ^ "_" ^ suffix)
        in
        let port : Units.Port.t =
          if String.equal name design.input
          then { data = i.s_tdata; valid = i.s_tvalid; ready = net "tready" 1 }
          else if String.equal name design.output
          then { data = net "tdata" width; valid = net "tvalid" 1; ready = i.m_tready }
          else
            { data = net "tdata" width; valid = net "tvalid" 1; ready = net "tready" 1 }
        in
        name, port)
    in
    let port name =
      match List.Assoc.find ports name ~equal:String.equal with
      | Some port -> port
      | None -> raise_s [%message "design.json: unknown stream" name]
    in
    List.iter design.nodes ~f:(fun node ->
      let (module Unit) = Units.of_node node in
      Unit.connect
        scope
        clocking
        ~inputs:(List.map node.inputs ~f:port)
        ~outputs:(List.map node.outputs ~f:port));
    let source = port design.input in
    let sink = port design.output in
    { O.s_tready = source.ready; m_tdata = sink.data; m_tvalid = sink.valid }
  ;;

  let hierarchical ?instance scope i =
    let module H = Hierarchy.In_scope (I) (O) in
    H.hierarchical ?instance ~scope ~name:design.top create i
  ;;
end

let verilog design =
  let module T =
    Make (struct
      let design = design
    end)
  in
  let module C = Circuit.With_interface (T.I) (T.O) in
  let scope = Scope.create ~auto_label_hierarchical_ports:true () in
  let circuit = C.create_exn ~name:design.top (T.create scope) in
  Rtl.create ~database:(Scope.circuit_database scope) Verilog [ circuit ]
  |> Rtl.full_hierarchy
  |> Rope.to_string
;;
