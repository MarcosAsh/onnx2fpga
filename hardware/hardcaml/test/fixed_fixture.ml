open! Core

(* Weights are listed channel by channel, [c][k] flattened. *)
type layer =
  { weights : int list
  ; bias : int list
  ; mult : int list
  ; out_min : int
  }

let fc0 =
  { weights = [ 3; -7; 12; 5; -1; 9 ]
  ; bias = [ 100; -50; 7 ]
  ; mult = [ 40; 90; 127 ]
  ; out_min = 0
  }
;;

let fc1 =
  { weights = [ -4; 6; 2; 11; -8; 1 ]
  ; bias = [ -3; 20 ]
  ; mult = [ 70; 33 ]
  ; out_min = -128
  }
;;

let dense ~name ~input ~output { weights; bias; mult; out_min } =
  let ints values = String.concat ~sep:", " (List.map values ~f:Int.to_string) in
  let mh = List.length bias in
  let mw = List.length weights / mh in
  sprintf
    {|{"name": "%s", "instance": "u_%s", "kind": "otf_dense",
       "params": {"MW": %d, "MH": %d, "ACT_BITS": 8, "WGT_BITS": 8, "ACC_BITS": 18,
                  "OUT_BITS": 8, "MULT_BITS": 8, "SHIFT": 6, "OUT_MIN": %d,
                  "OUT_MAX": 127, "OUT_ZP": 0},
       "memories": {"weights": {"elem_bits": 8, "lanes": %d, "values": [%s]},
                    "bias": {"elem_bits": 18, "lanes": 1, "values": [%s]},
                    "mult": {"elem_bits": 8, "lanes": 1, "values": [%s]}},
       "inputs": ["%s"], "outputs": ["%s"]}|}
    name
    name
    mw
    mh
    out_min
    mw
    (ints (List.map weights ~f:(fun w -> w land 0xff)))
    (ints (List.map bias ~f:(fun b -> b land 0x3ffff)))
    (ints mult)
    input
    output
;;

let description ~latency =
  sprintf
    {|{"format": 1, "top": "otf_top", "mode": "fixed", "latency": %d,
       "input": "x", "output": "y",
       "streams": [{"name": "x", "width": 16}, {"name": "h", "width": 24},
                   {"name": "y", "width": 16}],
       "nodes": [%s, %s]}|}
    latency
    (dense ~name:"fc0" ~input:"x" ~output:"h" fc0)
    (dense ~name:"fc1" ~input:"h" ~output:"y" fc1)
;;

let layer x { weights; bias; mult; out_min } =
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
