open! Core
open! Hardcaml
open! Signal

module Op = struct
  type t =
    | Arg_max
    | Arg_min
    | Max
    | Min
    | Sum
  [@@deriving sexp_of, enumerate]
end

module Stages = struct
  type t =
    { tree : int
    ; rescale : int
    }
  [@@deriving sexp_of]

  let none = { tree = 0; rescale = 0 }
end

module type Config = sig
  val lanes : int
  val bits : int
  val op : Op.t
  val last : bool
  val zero_point : int
  val mult : int
  val mult_bits : int
  val acc_bits : int
  val shift : int
  val out_zp : int
  val out_min : int
  val out_max : int
  val stages : Stages.t
end

let tree clocking ~cuts ~f items =
  let rec level index = function
    | [ item ] -> item
    | items ->
      let pairs = List.map (List.chunks_of items ~length:2) ~f:(reduce ~f) in
      if List.mem cuts index ~equal:Int.equal
      then level (index + 1) (List.map pairs ~f:(Clocking.reg_no_clear clocking))
      else level (index + 1) pairs
  in
  level 1 items
;;

let cuts ~levels ~stages = List.init stages ~f:(fun i -> (i + 1) * levels / (stages + 1))

module Make (Config : Config) = struct
  open Config

  let index_bits = Int.max 1 (Int.ceil_log2 lanes)

  let out_bits =
    match op with
    | Arg_max | Arg_min -> index_bits
    | Max | Min | Sum -> bits
  ;;

  let sum =
    match op with
    | Sum -> true
    | Arg_max | Arg_min | Max | Min -> false
  ;;

  (* The sum's tree also adds the constant -lanes * zero_point. *)
  let leaves = if sum then lanes + 1 else lanes
  let levels = Int.ceil_log2 leaves
  let latency = if sum then 2 + stages.tree + stages.rescale else 1 + stages.tree

  let () =
    if lanes < 1 || bits < 2 || (sum && acc_bits <= bits)
    then
      raise_s
        [%message
          "BUG: otf_reduce needs lanes and a wide enough sum"
            (lanes : int)
            (bits : int)
            (acc_bits : int)];
    if stages.tree < 0
       || (stages.tree > 0 && stages.tree >= levels)
       || stages.rescale < 0
       || stages.rescale > 3
       || ((not sum) && stages.rescale > 0)
    then raise_s [%message "BUG: otf_reduce stages out of range" (stages : Stages.t)]
  ;;

  module I = struct
    type 'a t =
      { clocking : 'a Clocking.t
      ; x : 'a With_valid.t [@bits lanes * bits]
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t = { y : 'a With_valid.t [@bits out_bits] } [@@deriving hardcaml]
  end

  module Requant = Otf_requant.Make_pipelined (struct
      let stages = stages.rescale
      let acc_bits = acc_bits
      let mult_bits = mult_bits
      let shift = shift
      let out_bits = bits
      let out_min = out_min
      let out_max = out_max
      let out_zp = out_zp
    end)

  (* Ties keep the left operand, the lower lanes, unless [last] asks for the later one. *)
  let pick ~larger a b =
    let va = sel_bottom a ~width:bits in
    let vb = sel_bottom b ~width:bits in
    match larger, last with
    | true, false -> mux2 (va >=+ vb) a b
    | true, true -> mux2 (vb >=+ va) b a
    | false, false -> mux2 (va <=+ vb) a b
    | false, true -> mux2 (vb <=+ va) b a
  ;;

  let create scope (i : Signal.t I.t) =
    let codes = split_lsb ~part_width:bits i.x.value in
    let tree = tree i.clocking ~cuts:(cuts ~levels ~stages:stages.tree) in
    let reg = Clocking.reg_no_clear i.clocking in
    let indexed =
      List.mapi codes ~f:(fun k code -> of_unsigned_int ~width:index_bits k @: code)
    in
    let%hw out =
      match op with
      | Arg_max -> sel_top (reg (tree ~f:(pick ~larger:true) indexed)) ~width:index_bits
      | Arg_min -> sel_top (reg (tree ~f:(pick ~larger:false) indexed)) ~width:index_bits
      | Max -> reg (tree ~f:(pick ~larger:true) codes)
      | Min -> reg (tree ~f:(pick ~larger:false) codes)
      | Sum ->
        let bias = of_signed_int ~width:acc_bits (-lanes * zero_point) in
        let%hw acc =
          reg (tree ~f:( +: ) (List.map codes ~f:(sresize ~width:acc_bits) @ [ bias ]))
        in
        reg
          (Requant.hierarchical
             scope
             { clocking = i.clocking; acc; mult = of_signed_int ~width:mult_bits mult })
            .out
    in
    let%hw valid = Clocking.pipeline i.clocking ~n:latency i.x.valid in
    { O.y = { valid; value = out } }
  ;;

  let hierarchical ?instance scope i =
    let module H = Hierarchy.In_scope (I) (O) in
    H.hierarchical ?instance ~scope ~name:"otf_reduce" create i
  ;;
end
