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
    let result, bank =
      List.fold
        units
        ~init:
          ( { With_valid.valid = i.features_valid; value = i.features }
          , Option.map banks ~f:(fun banks -> banks.bank) )
        ~f:(fun (x, bank) (module Unit : Units.Fixed_instance) ->
          Unit.connect scope clocking ~store ~bank x)
    in
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
