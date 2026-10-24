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

let drive (port : Port.t) ~data ~valid =
  port.data <-- data;
  port.valid <-- valid
;;

let only (ports : Port.t list) ~what node =
  match ports with
  | [ port ] -> port
  | _ ->
    raise_s [%message "design.json: unit needs exactly one stream" what (node : string)]
;;

(* Units with one stream in and one out all wire up the same way. *)
let single
  (node : Design.Node.t)
  ~(inputs : Port.t list)
  ~(outputs : Port.t list)
  ~(f :
      data:Signal.t -> valid:Signal.t -> ready:Signal.t -> Signal.t * Signal.t * Signal.t)
  =
  let s = only inputs ~what:"input" node.name in
  let m = only outputs ~what:"output" node.name in
  let ready, data, valid = f ~data:s.data ~valid:s.valid ~ready:m.ready in
  s.ready <-- ready;
  drive m ~data ~valid
;;

let of_node (node : Design.Node.t) : (module Instance) =
  let param = Design.Node.param node in
  let words = Design.Node.words node in
  let instance = node.instance in
  match node.kind with
  | "otf_fifo" ->
    (module struct
      module M = Otf_fifo.Make (struct
          let width = param "WIDTH"
          let depth = param "DEPTH"
        end)

      let circuit scope ~name =
        let module C = Circuit.With_interface (M.I) (M.O) in
        C.create_exn ~name (M.hierarchical scope)
      ;;

      let connect scope clocking ~inputs ~outputs =
        single node ~inputs ~outputs ~f:(fun ~data ~valid ~ready ->
          let o =
            M.hierarchical
              ~instance
              scope
              { clocking; s = { data; valid }; m = { ready } }
          in
          o.s.ready, o.m.data, o.m.valid)
      ;;
    end)
  | "otf_act" ->
    (module struct
      module M = Otf_act.Make (struct
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
        end)

      let circuit scope ~name =
        let module C = Circuit.With_interface (M.I) (M.O) in
        C.create_exn ~name (M.hierarchical scope)
      ;;

      let connect scope clocking ~inputs ~outputs =
        single node ~inputs ~outputs ~f:(fun ~data ~valid ~ready ->
          let o =
            M.hierarchical
              ~instance
              scope
              { clocking; s = { data; valid }; m = { ready } }
          in
          o.s.ready, o.m.data, o.m.valid)
      ;;
    end)
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

      let circuit scope ~name =
        let module C = Circuit.With_interface (M.I) (M.O) in
        C.create_exn ~name (M.hierarchical scope)
      ;;

      let connect scope clocking ~inputs ~outputs =
        match inputs with
        | [ (a : Port.t); (b : Port.t) ] ->
          let m = only outputs ~what:"output" node.name in
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

      let circuit scope ~name =
        let module C = Circuit.With_interface (M.I) (M.O) in
        C.create_exn ~name (M.hierarchical scope)
      ;;

      let connect scope clocking ~inputs ~(outputs : Port.t list) =
        let s = only inputs ~what:"input" node.name in
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
    (module struct
      module M = Otf_dwc.Make (struct
          let in_elems = param "IN_ELEMS"
          let out_elems = param "OUT_ELEMS"
          let elem_bits = param "ELEM_BITS"
          let total_elems = param "TOTAL_ELEMS"
        end)

      let circuit scope ~name =
        let module C = Circuit.With_interface (M.I) (M.O) in
        C.create_exn ~name (M.hierarchical scope)
      ;;

      let connect scope clocking ~inputs ~outputs =
        single node ~inputs ~outputs ~f:(fun ~data ~valid ~ready ->
          let o =
            M.hierarchical
              ~instance
              scope
              { clocking; s = { data; valid }; m = { ready } }
          in
          o.s.ready, o.m.data, o.m.valid)
      ;;
    end)
  | "otf_pool" ->
    (module struct
      module M = Otf_pool.Make (struct
          let channels = param "CHANNELS"
          let pe = param "PE"
          let window = param "WINDOW"
          let data_bits = param "DATA_BITS"
          let data_min = param "DATA_MIN"
        end)

      let circuit scope ~name =
        let module C = Circuit.With_interface (M.I) (M.O) in
        C.create_exn ~name (M.hierarchical scope)
      ;;

      let connect scope clocking ~inputs ~outputs =
        single node ~inputs ~outputs ~f:(fun ~data ~valid ~ready ->
          let o =
            M.hierarchical
              ~instance
              scope
              { clocking; s = { data; valid }; m = { ready } }
          in
          o.s.ready, o.m.data, o.m.valid)
      ;;
    end)
  | "otf_mvau" ->
    (module struct
      module M = Otf_mvau.Make (struct
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
          let weights = words "weights"
          let bias = words "bias"
          let mult = words "mult"
        end)

      let circuit scope ~name =
        let module C = Circuit.With_interface (M.I) (M.O) in
        C.create_exn ~name (M.hierarchical scope)
      ;;

      let connect scope clocking ~inputs ~outputs =
        single node ~inputs ~outputs ~f:(fun ~data ~valid ~ready ->
          let o =
            M.hierarchical
              ~instance
              scope
              { clocking; s = { data; valid }; m = { ready } }
          in
          o.s.ready, o.m.data, o.m.valid)
      ;;
    end)
  | "otf_swu" ->
    (module struct
      module M = Otf_swu.Make (struct
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

      let circuit scope ~name =
        let module C = Circuit.With_interface (M.I) (M.O) in
        C.create_exn ~name (M.hierarchical scope)
      ;;

      let connect scope clocking ~inputs ~outputs =
        single node ~inputs ~outputs ~f:(fun ~data ~valid ~ready ->
          let o =
            M.hierarchical
              ~instance
              scope
              { clocking; s = { data; valid }; m = { ready } }
          in
          o.s.ready, o.m.data, o.m.valid)
      ;;
    end)
  | "otf_swu_line" ->
    (module struct
      module M = Otf_swu_line.Make (struct
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
        end)

      let circuit scope ~name =
        let module C = Circuit.With_interface (M.I) (M.O) in
        C.create_exn ~name (M.hierarchical scope)
      ;;

      let connect scope clocking ~inputs ~outputs =
        single node ~inputs ~outputs ~f:(fun ~data ~valid ~ready ->
          let o =
            M.hierarchical
              ~instance
              scope
              { clocking; s = { data; valid }; m = { ready } }
          in
          o.s.ready, o.m.data, o.m.valid)
      ;;
    end)
  | kind -> raise_s [%message "design.json: no Hardcaml unit for" kind node.name]
;;

module type Fixed_instance = sig
  val latency : int

  val connect
    :  Scope.t
    -> Signal.t Clocking.t
    -> Signal.t With_valid.t
    -> Signal.t With_valid.t

  val circuit : Scope.t -> name:string -> Circuit.t
end

let of_fixed_node (node : Design.Node.t) : (module Fixed_instance) =
  let param = Design.Node.param node in
  let words = Design.Node.words node in
  let instance = node.instance in
  match node.kind with
  | "otf_dense" ->
    (module struct
      module M = Otf_dense.Make (struct
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
          let weights = words "weights"
          let bias = words "bias"
          let mult = words "mult"
        end)

      let latency = M.latency

      let circuit scope ~name =
        let module C = Circuit.With_interface (M.I) (M.O) in
        C.create_exn ~name (M.hierarchical scope)
      ;;

      let connect scope clocking x = (M.hierarchical ~instance scope { clocking; x }).y
    end)
  | kind ->
    raise_s [%message "design.json: no fixed latency Hardcaml unit for" kind node.name]
;;
