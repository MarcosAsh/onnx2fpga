open! Core
open! Hardcaml

(* Weights are listed channel by channel, [c][k] flattened; fc1 uses constant arithmetic. *)
type layer =
  { weights : int list
  ; bias : int list
  ; mult : int list
  ; out_min : int
  ; constant : bool
  }

let fc0 =
  { weights = [ 3; -7; 12; 5; -1; 9 ]
  ; bias = [ 100; -50; 7 ]
  ; mult = [ 40; 90; 127 ]
  ; out_min = 0
  ; constant = false
  }
;;

let fc1 =
  { weights = [ -4; 6; 2; 11; -8; 1 ]
  ; bias = [ -3; 20 ]
  ; mult = [ 70; 33 ]
  ; out_min = -128
  ; constant = true
  }
;;

(* The same two layers retrained: what a reload writes into the idle bank. *)
let fc0' =
  { weights = [ -9; 4; 1; -6; 8; 2 ]
  ; bias = [ -70; 33; 0 ]
  ; mult = [ 55; 21; 99 ]
  ; out_min = 0
  ; constant = false
  }
;;

let fc1' =
  { weights = [ 7; -3; -12; 5; 0; 10 ]
  ; bias = [ 15; -40 ]
  ; mult = [ 64; 120 ]
  ; out_min = -128
  ; constant = false
  }
;;

let dense ~name ~input ~output ~reloadable { weights; bias; mult; out_min; constant } =
  let ints values = String.concat ~sep:", " (List.map values ~f:Int.to_string) in
  let mh = List.length bias in
  let mw = List.length weights / mh in
  sprintf
    {|{"name": "%s", "instance": "u_%s", "kind": "otf_dense",
       "params": {"MW": %d, "MH": %d, "ACT_BITS": 8, "WGT_BITS": 8, "ACC_BITS": 18,
                  "OUT_BITS": 8, "MULT_BITS": 8, "SHIFT": 6, "OUT_MIN": %d,
                  "OUT_MAX": 127, "OUT_ZP": 0, "CONSTANT": %d, "RELOADABLE": %d},
       "memories": {"weights": {"elem_bits": 8, "lanes": %d, "values": [%s]},
                    "bias": {"elem_bits": 18, "lanes": 1, "values": [%s]},
                    "mult": {"elem_bits": 8, "lanes": 1, "values": [%s]}},
       "inputs": ["%s"], "outputs": ["%s"]}|}
    name
    name
    mw
    mh
    out_min
    (Bool.to_int (constant && not reloadable))
    (Bool.to_int reloadable)
    mw
    (ints (List.map weights ~f:(fun w -> w land 0xff)))
    (ints (List.map bias ~f:(fun b -> b land 0x3ffff)))
    (ints mult)
    input
    output
;;

let design ~latency ~reloadable =
  sprintf
    {|{"format": 1, "top": "otf_top", "mode": "fixed", "latency": %d,
       "input": "x", "output": "y",
       "streams": [{"name": "x", "width": 16}, {"name": "h", "width": 24},
                   {"name": "y", "width": 16}],
       "nodes": [%s, %s]}|}
    latency
    (dense ~name:"fc0" ~input:"x" ~output:"h" ~reloadable fc0)
    (dense ~name:"fc1" ~input:"h" ~output:"y" ~reloadable fc1)
;;

let description ~latency = design ~latency ~reloadable:false
let reloadable = design ~latency:6 ~reloadable:true

let layer x { weights; bias; mult; out_min; constant = _ } =
  List.mapi bias ~f:(fun c b ->
    let dot =
      List.foldi x ~init:b ~f:(fun k sum a ->
        sum + (a * List.nth_exn weights ((c * List.length x) + k)))
    in
    Model.requant
      ~acc_bits:18
      ~mult_bits:8
      ~shift:6
      ~out_min
      ~out_max:127
      ~out_zp:0
      ~acc:dot
      ~mult:(List.nth_exn mult c))
;;

let model x = layer (layer x fc0) fc1

(* The table halves a code; the gate multiplies it into the two layers' answer. *)
let half code = code asr 1

let gated =
  let table =
    String.concat
      ~sep:", "
      (List.init 256 ~f:(fun k -> Int.to_string (half (k - 128) land 0xff)))
  in
  sprintf
    {|{"format": 1, "top": "otf_top", "mode": "fixed", "latency": 8,
       "input": "x", "output": "y",
       "streams": [{"name": "x", "width": 16}, {"name": "h", "width": 24},
                   {"name": "m", "width": 16}, {"name": "t", "width": 16},
                   {"name": "y", "width": 16}],
       "nodes": [%s, %s,
         {"name": "half", "instance": "u_half", "kind": "otf_table",
          "params": {"LANES": 2, "IN_BITS": 8, "OUT_BITS": 8},
          "memories": {"tables": {"elem_bits": 8, "lanes": 1, "values": [%s]},
                       "lane_table": {"elem_bits": 1, "lanes": 1, "values": [0, 0]}},
          "inputs": ["x"], "outputs": ["t"]},
         {"name": "gate", "instance": "u_gate", "kind": "otf_eltwise",
          "params": {"LANES": 2, "BITS": 8, "OP": 1, "OPERANDS": 2, "PORT0": 0,
                     "PORT1": 1, "ZP0": 0, "ZP1": 0, "MULT0": 1, "MULT1": 1,
                     "MULT": 300, "MULT_BITS": 10, "ACC_BITS": 18, "SHIFT": 12,
                     "OUT_ZP": 3, "OUT_MIN": -128, "OUT_MAX": 127},
          "memories": {}, "inputs": ["m", "t"], "outputs": ["y"]}]}|}
    (dense ~name:"fc0" ~input:"x" ~output:"h" ~reloadable:false fc0)
    (dense ~name:"fc1" ~input:"h" ~output:"m" ~reloadable:false fc1)
    table
;;

let gated_model x =
  List.map2_exn (model x) x ~f:(fun m code ->
    Model.python_requant
      ~shift:12
      ~out_min:(-128)
      ~out_max:127
      ~out_zp:3
      ~acc:(m * half code)
      ~mult:300)
;;

let reloaded_model x = layer (layer x fc0') fc1'

let image layers =
  List.concat_map layers ~f:(fun { weights; bias; mult; _ } ->
    Model.words_of_ints ~bits:8 weights
    @ Model.words_of_ints ~bits:18 bias
    @ Model.words_of_ints ~bits:8 mult)
  |> List.map ~f:(Bits.uresize ~width:32)
;;

let compiled_image = image [ fc0; fc1 ]
let reloaded_image = image [ fc0'; fc1' ]
