open! Core
open! Hardcaml
open! Signal

module type Config = sig
  val design : Design.t
end

module Make (Config : Config) = struct
  open Config

  let stated_latency =
    match design.mode with
    | Fixed { latency } -> latency
    | Streaming ->
      raise_s [%message "BUG: Fixed_top needs a fixed latency design" design.top]
  ;;

  let units = List.map design.nodes ~f:Units.of_fixed_node

  let latency =
    List.sum (module Int) units ~f:(fun (module Unit : Units.Fixed_instance) ->
      Unit.latency)
  ;;

  (* The compiler and the units must agree on the latency, or the manifest would lie. *)
  let () =
    if latency <> stated_latency
    then
      raise_s
        [%message
          "design.json: the stated latency is not the sum over the units"
            (stated_latency : int)
            (latency : int)];
    let tail =
      List.fold design.nodes ~init:design.input ~f:(fun source (node : Design.Node.t) ->
        match node.inputs, node.outputs with
        | [ input ], [ output ] when String.equal input source -> output
        | _ ->
          raise_s
            [%message "design.json: fixed latency nodes must form a chain" node.name])
    in
    if not (String.equal tail design.output)
    then raise_s [%message "design.json: the chain does not end at the output" tail]
  ;;

  module I = struct
    type 'a t =
      { clock : 'a
      ; clear : 'a
      ; features : 'a [@bits Design.width design design.input]
      ; features_valid : 'a
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t =
      { prediction : 'a [@bits Design.width design design.output]
      ; prediction_valid : 'a
      }
    [@@deriving hardcaml]
  end

  let create (scope : Scope.t) (i : Signal.t I.t) =
    let clocking = { Clocking.clock = i.clock; clear = i.clear } in
    let result =
      List.fold
        units
        ~init:{ With_valid.valid = i.features_valid; value = i.features }
        ~f:(fun x (module Unit : Units.Fixed_instance) -> Unit.connect scope clocking x)
    in
    { O.prediction = result.value; prediction_valid = result.valid }
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
