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

  let units =
    List.folding_map design.nodes ~init:0 ~f:(fun base node ->
      let (module Unit : Units.Fixed_instance) = Units.of_fixed_node ~base node in
      base + Unit.words, (module Unit : Units.Fixed_instance))
  ;;

  let image =
    List.concat_map units ~f:(fun (module Unit : Units.Fixed_instance) -> Unit.image)
  ;;

  let words = List.length image
  let reloadable = words > 0

  (* When each stream is ready: a unit starts once its latest input arrives. *)
  let arrivals =
    List.fold
      (List.zip_exn design.nodes units)
      ~init:(String.Map.singleton design.input 0)
      ~f:(fun arrival ((node : Design.Node.t), (module Unit : Units.Fixed_instance)) ->
        let start =
          List.map node.inputs ~f:(fun input ->
            match Map.find arrival input with
            | Some time -> time
            | None ->
              raise_s
                [%message
                  "design.json: a node reads a stream no earlier node writes"
                    node.name
                    input])
          |> List.max_elt ~compare:Int.compare
          |> Option.value_exn ~message:("design.json: node has no inputs " ^ node.name)
        in
        match node.outputs with
        | [ output ] -> Map.set arrival ~key:output ~data:(start + Unit.latency)
        | _ ->
          raise_s [%message "design.json: fixed latency nodes have one output" node.name])
  ;;

  let latency =
    match Map.find arrivals design.output with
    | Some time -> time
    | None -> raise_s [%message "design.json: no node writes the output" design.output]
  ;;

  (* The compiler and the units must agree on the latency, or the manifest would lie. *)
  let () =
    if latency <> stated_latency
    then
      raise_s
        [%message
          "design.json: the stated latency is not the longest path through the units"
            (stated_latency : int)
            (latency : int)]
  ;;

  module I = struct
    type 'a t =
      { clock : 'a
      ; clear : 'a
      ; features : 'a [@bits Design.width design design.input]
      ; features_valid : 'a
      ; weights_write : 'a option [@exists reloadable]
      ; weights_address : 'a option [@bits Weight_banks.address_bits] [@exists reloadable]
      ; weights_data : 'a option [@bits 32] [@exists reloadable]
      ; weights_commit : 'a option [@exists reloadable]
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t =
      { prediction : 'a [@bits Design.width design design.output]
      ; prediction_valid : 'a
      ; prediction_bank : 'a option [@exists reloadable]
      ; weights_bank : 'a option [@exists reloadable]
      ; weights_version : 'a option [@bits 32] [@exists reloadable]
      ; weights_crc : 'a option [@bits 32] [@exists reloadable]
      ; weights_idle_crc : 'a option [@bits 32] [@exists reloadable]
      }
    [@@deriving hardcaml]
  end

  module Banks = Weight_banks.Make (struct
      let image = image
    end)

  let create (scope : Scope.t) (i : Signal.t I.t) =
    let clocking = { Clocking.clock = i.clock; clear = i.clear } in
    let banks, store =
      match i.weights_write, i.weights_address, i.weights_data, i.weights_commit with
      | Some write, Some address, Some data, Some commit ->
        let write = write &: ~:commit in
        let banks =
          Banks.hierarchical scope { clock = i.clock; write; address; data; commit }
        in
        ( Some banks
        , Some { Weight_banks.Write.write; bank = ~:(banks.bank); address; data } )
      | _ -> None, None
    in
    (* Earlier inputs of a unit wait in plain registers until its latest one arrives. *)
    let streams =
      List.fold
        (List.zip_exn design.nodes units)
        ~init:
          (String.Map.singleton
             design.input
             ( { With_valid.valid = i.features_valid; value = i.features }
             , 0
             , Option.map banks ~f:(fun banks -> banks.bank) ))
        ~f:(fun streams ((node : Design.Node.t), (module Unit : Units.Fixed_instance)) ->
          let inputs = List.map node.inputs ~f:(Map.find_exn streams) in
          let start =
            List.fold inputs ~init:0 ~f:(fun latest (_, time, _) -> Int.max latest time)
          in
          let aligned =
            List.map inputs ~f:(fun ((x : Signal.t With_valid.t), time, bank) ->
              let n = start - time in
              ( { With_valid.valid = Clocking.pipeline clocking ~n x.valid
                ; value = Clocking.pipeline_no_clear clocking ~n x.value
                }
              , Option.map bank ~f:(Clocking.pipeline_no_clear clocking ~n) ))
          in
          let y, bank =
            Unit.connect
              scope
              clocking
              ~store
              ~bank:(snd (List.hd_exn aligned))
              (List.map aligned ~f:fst)
          in
          Map.set
            streams
            ~key:(List.hd_exn node.outputs)
            ~data:(y, start + Unit.latency, bank))
    in
    let result, _, bank = Map.find_exn streams design.output in
    let field f = Option.map banks ~f in
    { O.prediction = result.value
    ; prediction_valid = result.valid
    ; prediction_bank = bank
    ; weights_bank = field (fun banks -> banks.bank)
    ; weights_version = field (fun banks -> banks.version)
    ; weights_crc = field (fun banks -> banks.crc)
    ; weights_idle_crc = field (fun banks -> banks.idle_crc)
    }
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
