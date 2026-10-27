open! Core
open! Hardcaml
open! Signal

module Port = struct
  type t =
    { data : Signal.t
    ; valid : Signal.t
    ; ready : Signal.t
    }
end

module type Instance = sig
  val connect
    :  Scope.t
    -> Signal.t Clocking.t
    -> inputs:Port.t list
    -> outputs:Port.t list
    -> unit

  val circuit : Scope.t -> name:string -> Circuit.t
end

module type Stream_unit = sig
  module In : sig
    type 'a t =
      { data : 'a
      ; valid : 'a
      }
  end

  module Out : sig
    type 'a t =
      { data : 'a
      ; valid : 'a
      }
  end

  module I : sig
    type 'a t =
      { clocking : 'a Clocking.t
      ; s : 'a In.t
      ; m : 'a Stream.Dest.t
      }
    [@@deriving hardcaml]
  end

  module O : sig
    type 'a t =
      { s : 'a Stream.Dest.t
      ; m : 'a Out.t
      }
    [@@deriving hardcaml]
  end

  val hierarchical : ?instance:string -> Scope.t -> Signal.t I.t -> Signal.t O.t
end

module Circuit_of (M : sig
    module I : Interface.S
    module O : Interface.S

    val hierarchical : ?instance:string -> Scope.t -> Signal.t I.t -> Signal.t O.t
  end) =
struct
  let circuit scope ~name =
    let module C = Circuit.With_interface (M.I) (M.O) in
    C.create_exn ~name (M.hierarchical scope)
  ;;
end

let drive (port : Port.t) ~data ~valid =
  port.data <-- data;
  port.valid <-- valid
;;

let only (ports : Port.t list) ~what (node : Design.Node.t) =
  match ports with
  | [ port ] -> port
  | _ -> raise_s [%message "design.json: unit needs exactly one stream" what node.name]
;;

(* Units with one stream in and one out all wire up the same way. *)
let stream (node : Design.Node.t) (module M : Stream_unit) : (module Instance) =
  (module struct
    include Circuit_of (M)

    let connect scope clocking ~inputs ~outputs =
      let s = only inputs ~what:"input" node in
      let m = only outputs ~what:"output" node in
      let o =
        M.hierarchical
          ~instance:node.instance
          scope
          { clocking; s = { data = s.data; valid = s.valid }; m = { ready = m.ready } }
      in
      s.ready <-- o.s.ready;
      drive m ~data:o.m.data ~valid:o.m.valid
    ;;
  end)
;;

let of_node (node : Design.Node.t) : (module Instance) =
  let param = Design.Node.param node in
  let words = Design.Node.words node in
  let instance = node.instance in
  match node.kind with
  | "otf_fifo" ->
    stream
      node
      (module struct
        include Otf_fifo.Make (struct
            let width = param "WIDTH"
            let depth = param "DEPTH"
          end)

        module In = Beat
        module Out = Beat
      end)
  | "otf_act" ->
    stream
      node
      (module Otf_act.Make (struct
          let channels = param "CHANNELS"
          let pe = param "PE"
          let in_bits = param "IN_BITS"
          let out_bits = param "OUT_BITS"
          let mult_bits = param "MULT_BITS"
          let shift = param "SHIFT"
          let out_min = param "OUT_MIN"
          let out_max = param "OUT_MAX"
          let in_zp = param "IN_ZP"
          let out_zp = param "OUT_ZP"
          let mult = words "mult"
        end))
  | "otf_add" ->
    (module struct
      module M = Otf_add.Make (struct
          let pe = param "PE"
          let in_bits = param "IN_BITS"
          let out_bits = param "OUT_BITS"
          let acc_bits = param "ACC_BITS"
          let left_mult = param "LEFT_MULT"
          let right_mult = param "RIGHT_MULT"
          let shift = param "SHIFT"
          let out_min = param "OUT_MIN"
          let out_max = param "OUT_MAX"
          let bias = param "BIAS"
          let out_zp = param "OUT_ZP"
        end)

      include Circuit_of (M)

      let connect scope clocking ~inputs ~outputs =
        match inputs with
        | [ (a : Port.t); (b : Port.t) ] ->
          let m = only outputs ~what:"output" node in
          let o =
            M.hierarchical
              ~instance
              scope
              { clocking
              ; a = { data = a.data; valid = a.valid }
              ; b = { data = b.data; valid = b.valid }
              ; m = { ready = m.ready }
              }
          in
          a.ready <-- o.a.ready;
          b.ready <-- o.b.ready;
          drive m ~data:o.m.data ~valid:o.m.valid
        | _ -> raise_s [%message "design.json: otf_add needs two streams" node.name]
      ;;
    end)
  | "otf_dup" ->
    (module struct
      module M = Otf_dup.Make (struct
          let width = param "WIDTH"
          let fanout = param "FANOUT"
        end)

      include Circuit_of (M)

      let connect scope clocking ~inputs ~(outputs : Port.t list) =
        let s = only inputs ~what:"input" node in
        if List.length outputs <> param "FANOUT"
        then
          raise_s
            [%message "design.json: otf_dup fanout differs from its outputs" node.name];
        let o =
          M.hierarchical
            ~instance
            scope
            { clocking
            ; s = { data = s.data; valid = s.valid }
            ; m =
                List.map outputs ~f:(fun (m : Port.t) -> { Stream.Dest.ready = m.ready })
            }
        in
        s.ready <-- o.s.ready;
        List.iter2_exn outputs o.m ~f:(fun port branch ->
          drive port ~data:branch.data ~valid:branch.valid)
      ;;
    end)
  | "otf_dwc" ->
    stream
      node
      (module Otf_dwc.Make (struct
          let in_elems = param "IN_ELEMS"
          let out_elems = param "OUT_ELEMS"
          let elem_bits = param "ELEM_BITS"
          let total_elems = param "TOTAL_ELEMS"
        end))
  | "otf_pool" ->
    stream
      node
      (module struct
        include Otf_pool.Make (struct
            let channels = param "CHANNELS"
            let pe = param "PE"
            let window = param "WINDOW"
            let data_bits = param "DATA_BITS"
            let data_min = param "DATA_MIN"
          end)

        module In = Beat
        module Out = Beat
      end)
  | "otf_mvau" ->
    stream
      node
      (module Otf_mvau.Make (struct
          let mw = param "MW"
          let mh = param "MH"
          let simd = param "SIMD"
          let pe = param "PE"
          let act_bits = param "ACT_BITS"
          let wgt_bits = param "WGT_BITS"
          let acc_bits = param "ACC_BITS"
          let out_bits = param "OUT_BITS"
          let mult_bits = param "MULT_BITS"
          let shift = param "SHIFT"
          let out_min = param "OUT_MIN"
          let out_max = param "OUT_MAX"
          let out_zp = param "OUT_ZP"
          let block_ram = param "BLOCK_RAM" <> 0
          let weights = words "weights"
          let bias = words "bias"
          let mult = words "mult"
        end))
  | "otf_swu" ->
    stream
      node
      (module struct
        include Otf_swu.Make (struct
            let ifm_h = param "IFM_H"
            let ifm_w = param "IFM_W"
            let ifm_ch = param "IFM_CH"
            let ofm_h = param "OFM_H"
            let ofm_w = param "OFM_W"
            let kh = param "KH"
            let kw = param "KW"
            let stride_h = param "STRIDE_H"
            let stride_w = param "STRIDE_W"
            let pad_t = param "PAD_T"
            let pad_l = param "PAD_L"
            let dil_h = param "DIL_H"
            let dil_w = param "DIL_W"
            let simd = param "SIMD"
            let data_bits = param "DATA_BITS"
            let pad_value = param "PAD_VALUE"
          end)

        module In = Beat
        module Out = Beat
      end)
  | "otf_swu_line" ->
    stream
      node
      (module Otf_swu_line.Make (struct
          let ifm_h = param "IFM_H"
          let ifm_w = param "IFM_W"
          let ifm_ch = param "IFM_CH"
          let ofm_h = param "OFM_H"
          let ofm_w = param "OFM_W"
          let kh = param "KH"
          let kw = param "KW"
          let stride_h = param "STRIDE_H"
          let stride_w = param "STRIDE_W"
          let pad_t = param "PAD_T"
          let pad_l = param "PAD_L"
          let dil_h = param "DIL_H"
          let dil_w = param "DIL_W"
          let simd = param "SIMD"
          let par = param "PAR"
          let data_bits = param "DATA_BITS"
          let pad_value = param "PAD_VALUE"
        end))
  | kind -> raise_s [%message "design.json: no Hardcaml unit for" kind node.name]
;;

module type Fixed_instance = sig
  val latency : int
  val words : int
  val image : Bits.t list

  val connect
    :  Scope.t
    -> Signal.t Clocking.t
    -> store:Signal.t Weight_banks.Write.t option
    -> bank:Signal.t option
    -> Signal.t With_valid.t list
    -> Signal.t With_valid.t * Signal.t option

  val circuit : Scope.t -> name:string -> Circuit.t
end

(* Frozen units hold no bank words and only delay the tag. *)
module Frozen (M : sig
    val latency : int
  end) =
struct
  let words = 0
  let image = []

  let delay_bank clocking bank =
    Option.map bank ~f:(Clocking.pipeline_no_clear clocking ~n:M.latency)
  ;;
end

let single (node : Design.Node.t) = function
  | [ x ] -> x
  | _ -> raise_s [%message "design.json: unit needs exactly one input" node.name]
;;

let of_fixed_node ?(base = 0) (node : Design.Node.t) : (module Fixed_instance) =
  let param ?default name = Design.Node.param ?default node name in
  let words = Design.Node.words node in
  match node.kind with
  | "otf_dense" ->
    let module Config = struct
      let mw = param "MW"
      let mh = param "MH"
      let act_bits = param "ACT_BITS"
      let wgt_bits = param "WGT_BITS"
      let acc_bits = param "ACC_BITS"
      let out_bits = param "OUT_BITS"
      let mult_bits = param "MULT_BITS"
      let shift = param "SHIFT"
      let out_min = param "OUT_MIN"
      let out_max = param "OUT_MAX"
      let out_zp = param "OUT_ZP"
      let arithmetic = if param "CONSTANT" = 1 then `Constant else `Multiplier
      let weights = words "weights"
      let bias = words "bias"
      let mult = words "mult"

      let stages =
        { Otf_dense.Stages.product = param ~default:0 "PRODUCT_STAGES"
        ; sum = param ~default:0 "SUM_STAGES"
        ; rescale = param ~default:0 "RESCALE_STAGES"
        }
      ;;

      let base = base
    end
    in
    if param "RELOADABLE" = 1
    then
      (module struct
        module M = Otf_dense.Make_reloadable (Config)

        let latency = M.latency
        let words = M.words
        let image = M.image

        include Circuit_of (M)

        let connect scope clocking ~store ~bank inputs =
          let o =
            M.hierarchical
              ~instance:node.instance
              scope
              { clocking
              ; x = single node inputs
              ; x_bank = Option.value_exn bank
              ; store = Option.value_exn store
              }
          in
          o.y, Some o.y_bank
        ;;
      end)
    else
      (module struct
        module M = Otf_dense.Make (Config)

        let latency = M.latency

        include Frozen (M)
        include Circuit_of (M)

        let connect scope clocking ~store:_ ~bank inputs =
          ( (M.hierarchical
               ~instance:node.instance
               scope
               { clocking; x = single node inputs })
              .y
          , delay_bank clocking bank )
        ;;
      end)
  | "otf_trees" ->
    let ints name ~f = Array.of_list (List.map (words name) ~f) in
    let kind = ints "kind" ~f:Bits.to_unsigned_int in
    let feature = ints "feature" ~f:Bits.to_unsigned_int in
    let value = ints "value" ~f:Bits.to_signed_int in
    let if_true = ints "if_true" ~f:Bits.to_unsigned_int in
    let if_false = ints "if_false" ~f:Bits.to_unsigned_int in
    (* Kind 2 is a leaf, 1 an equality split and 0 a less or equal split. *)
    let rec tree index : Otf_trees.Tree.t =
      if kind.(index) = 2
      then Leaf value.(index)
      else
        Node
          { split =
              { feature = feature.(index)
              ; threshold = value.(index)
              ; equal = kind.(index) = 1
              }
          ; if_true = tree if_true.(index)
          ; if_false = tree if_false.(index)
          }
    in
    (module struct
      module M = Otf_trees.Make (struct
          let features = param "FEATURES"
          let feature_bits = param "FEATURE_BITS"
          let leaf_bits = param "LEAF_BITS"
          let sum_bits = param "SUM_BITS"

          (* Each root names the score it adds to, and each score has a bias. *)
          let scores =
            let roots =
              List.zip_exn
                (List.map (words "roots") ~f:Bits.to_unsigned_int)
                (List.map (words "scores") ~f:Bits.to_unsigned_int)
            in
            List.mapi (words "bias") ~f:(fun score bias ->
              { Otf_trees.Score.trees =
                  List.filter_map roots ~f:(fun (root, owner) ->
                    Option.some_if (owner = score) (tree root))
              ; bias = Bits.to_signed_int bias
              })
          ;;

          let decision = param "DECISION" = 1

          let stages =
            { Otf_trees.Stages.select = param ~default:0 "SELECT_STAGES"
            ; sum = param ~default:0 "SUM_STAGES"
            ; pick = param ~default:0 "PICK_STAGES"
            }
          ;;
        end)

      let latency = M.latency

      include Frozen (M)
      include Circuit_of (M)

      let connect scope clocking ~store:_ ~bank inputs =
        ( (M.hierarchical
             ~instance:node.instance
             scope
             { clocking; x = single node inputs })
            .y
        , delay_bank clocking bank )
      ;;
    end)
  | ("otf_table" | "otf_concat") as kind ->
    let size = 1 lsl param "IN_BITS" in
    (* A concat's inputs arrive aligned and sit side by side, the first in the low lanes. *)
    let join (inputs : Signal.t With_valid.t list) =
      if String.equal kind "otf_table"
      then single node inputs
      else (
        match inputs with
        | first :: _ ->
          { With_valid.valid = first.valid
          ; value = Signal.concat_lsb (List.map inputs ~f:(fun x -> x.value))
          }
        | [] -> raise_s [%message "design.json: otf_concat has no inputs" node.name])
    in
    (module struct
      module M = Otf_table.Make (struct
          let lanes = param "LANES"
          let in_bits = param "IN_BITS"
          let out_bits = param "OUT_BITS"

          let tables =
            List.map (words "tables") ~f:Bits.to_signed_int
            |> List.chunks_of ~length:size
            |> List.map ~f:Array.of_list
          ;;

          let lane_table = List.map (words "lane_table") ~f:Bits.to_unsigned_int
        end)

      let latency = M.latency

      include Frozen (M)
      include Circuit_of (M)

      let connect scope clocking ~store:_ ~bank inputs =
        ( (M.hierarchical ~instance:node.instance scope { clocking; x = join inputs }).y
        , delay_bank clocking bank )
      ;;
    end)
  | "otf_eltwise" ->
    let count = param "OPERANDS" in
    let each name = List.init count ~f:(fun k -> param (name ^ Int.to_string k)) in
    (module struct
      module M = Otf_eltwise.Make (struct
          let lanes = param "LANES"
          let bits = param "BITS"
          let op = List.nth_exn Otf_eltwise.Op.all (param "OP")

          (* A negative port is a constant operand, its codes in memory const<k>. *)
          let operands =
            List.mapi (each "PORT") ~f:(fun k port : Otf_eltwise.Operand.t ->
              if port >= 0
              then Port port
              else
                Constant
                  (List.map (words ("const" ^ Int.to_string k)) ~f:Bits.to_signed_int))
          ;;

          let zero_points = each "ZP"
          let mults = each "MULT"

          (* Max, Min and Where keep a table per port operand in memory table<k>. *)
          let tables =
            List.init count ~f:(fun k ->
              let name = "table" ^ Int.to_string k in
              if List.Assoc.mem node.memories name ~equal:String.equal
              then Array.of_list_map (words name) ~f:Bits.to_signed_int
              else [||])
          ;;

          let mult = param "MULT"
          let mult_bits = param "MULT_BITS"
          let acc_bits = param "ACC_BITS"
          let shift = param "SHIFT"
          let out_zp = param "OUT_ZP"
          let out_min = param "OUT_MIN"
          let out_max = param "OUT_MAX"

          let stages =
            { Otf_eltwise.Stages.product = param ~default:0 "PRODUCT_STAGES"
            ; rescale = param ~default:0 "RESCALE_STAGES"
            }
          ;;
        end)

      let latency = M.latency

      include Frozen (M)
      include Circuit_of (M)

      let connect scope clocking ~store:_ ~bank (inputs : Signal.t With_valid.t list) =
        match inputs with
        | first :: _ ->
          ( (M.hierarchical
               ~instance:node.instance
               scope
               { clocking
               ; valid = first.valid
               ; x = List.map inputs ~f:(fun (x : Signal.t With_valid.t) -> x.value)
               })
              .y
          , delay_bank clocking bank )
        | [] -> raise_s [%message "design.json: otf_eltwise has no inputs" node.name]
      ;;
    end)
  | "otf_reduce" ->
    (module struct
      module M = Otf_reduce.Make (struct
          let lanes = param "LANES"
          let bits = param "BITS"
          let op = List.nth_exn Otf_reduce.Op.all (param "OP")
          let last = param "LAST" = 1
          let zero_point = param "ZP"
          let mult = param "MULT"
          let mult_bits = param "MULT_BITS"
          let acc_bits = param "ACC_BITS"
          let shift = param "SHIFT"
          let out_zp = param "OUT_ZP"
          let out_min = param "OUT_MIN"
          let out_max = param "OUT_MAX"

          let stages =
            { Otf_reduce.Stages.tree = param ~default:0 "TREE_STAGES"
            ; rescale = param ~default:0 "RESCALE_STAGES"
            }
          ;;
        end)

      let latency = M.latency

      include Frozen (M)
      include Circuit_of (M)

      let connect scope clocking ~store:_ ~bank inputs =
        ( (M.hierarchical
             ~instance:node.instance
             scope
             { clocking; x = single node inputs })
            .y
        , delay_bank clocking bank )
      ;;
    end)
  | "otf_softmax" ->
    let ints ?(signed = true) name =
      if List.Assoc.mem node.memories name ~equal:String.equal
      then
        Array.of_list_map (words name) ~f:(fun word ->
          if signed then Bits.to_signed_int word else Bits.to_unsigned_int word)
      else [||]
    in
    (module struct
      module M = Otf_softmax.Make (struct
          let lanes = param "LANES"
          let in_bits = param "IN_BITS"
          let subtract_max = param "MAX" = 1

          (* MODE counts Divide, Log, Root and Max; older designs only say LOG. *)
          let mode =
            List.nth_exn
              [ Otf_softmax.Mode.Divide; Log; Root; Max ]
              (param ~default:(param "LOG") "MODE")
          ;;

          let sum_table = ints ~signed:false "sum_table"
          let sum_bits = param "SUM_BITS"
          let numer_table = ints "numer_table"
          let numer_bits = param "NUMER_BITS"
          let total_bits = param "TOTAL_BITS"
          let min_exp = param "MIN_EXP"
          let mantissa_bits = param "MANTISSA_BITS"
          let scale_table = ints "scale_table"
          let exp_table = ints "exp_table"
          let scale_bits = param "SCALE_BITS"
          let shift = param "SHIFT"
          let out_bits = param "OUT_BITS"
          let out_zp = param "OUT_ZP"
          let out_min = param "OUT_MIN"
          let out_max = param "OUT_MAX"

          let stages =
            { Otf_softmax.Stages.max = param ~default:0 "MAX_STAGES"
            ; table = param ~default:0 "TABLE_STAGES"
            ; sum = param ~default:0 "SUM_STAGES"
            ; normalize = param ~default:0 "NORMALIZE_STAGES"
            ; rescale = param ~default:0 "RESCALE_STAGES"
            }
          ;;
        end)

      let latency = M.latency

      include Frozen (M)
      include Circuit_of (M)

      let connect scope clocking ~store:_ ~bank inputs =
        ( (M.hierarchical
             ~instance:node.instance
             scope
             { clocking; x = single node inputs })
            .y
        , delay_bank clocking bank )
      ;;
    end)
  | kind ->
    raise_s [%message "design.json: no fixed latency Hardcaml unit for" kind node.name]
;;
