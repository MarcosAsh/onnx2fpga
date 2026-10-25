open! Core
open! Hardcaml
open! Signal

module Split = struct
  module T = struct
    type t =
      { feature : int
      ; threshold : int
      ; equal : bool
      }
    [@@deriving compare, sexp_of]
  end

  include T
  include Comparator.Make (T)
end

module Tree = struct
  type t =
    | Leaf of int
    | Node of
        { split : Split.t
        ; if_true : t
        ; if_false : t
        }
  [@@deriving sexp_of]

  let rec splits = function
    | Leaf _ -> []
    | Node { split; if_true; if_false } -> (split :: splits if_true) @ splits if_false
  ;;

  let rec leaves = function
    | Leaf value -> [ value ]
    | Node { if_true; if_false; _ } -> leaves if_true @ leaves if_false
  ;;

  let rec depth = function
    | Leaf _ -> 0
    | Node { if_true; if_false; _ } -> 1 + Int.max (depth if_true) (depth if_false)
  ;;

  let rec eval t x =
    match t with
    | Leaf value -> value
    | Node { split = { feature; threshold; equal }; if_true; if_false } ->
      let taken = if equal then x.(feature) = threshold else x.(feature) <= threshold in
      eval (if taken then if_true else if_false) x
  ;;
end

module Stages = struct
  type t =
    { select : int
    ; sum : int
    }
  [@@deriving sexp_of]

  let none = { select = 0; sum = 0 }
end

module type Config = sig
  val features : int
  val feature_bits : int
  val leaf_bits : int
  val sum_bits : int
  val trees : Tree.t list
  val bias : int
  val decision : bool
  val stages : Stages.t
end

(* Fan in 4 a stage, which is two binary levels: ceil (levels / 2). *)
let adder_stages operands = (Int.ceil_log2 operands + 1) / 2

module Make (Config : Config) = struct
  open Config

  let depth =
    List.fold trees ~init:0 ~f:(fun deepest tree -> Int.max deepest (Tree.depth tree))
  ;;

  let levels = Int.ceil_log2 (List.length trees + 1)
  let sum_registers = adder_stages (List.length trees + 1) + stages.sum
  let latency = 2 + stages.select + sum_registers

  (* Evenly spaced cuts: [count] of the boundaries strictly inside [span] levels. *)
  let spread ~count ~span = List.init count ~f:(fun i -> (i + 1) * span / (count + 1))

  (* Leaf select cuts are node depths; adder tree cuts are levels. *)
  let select_cuts = spread ~count:stages.select ~span:depth
  let sum_cuts = spread ~count:(sum_registers - 1) ~span:levels @ [ levels ]

  let () =
    let fits ~bits value = value >= -(1 lsl (bits - 1)) && value < 1 lsl (bits - 1) in
    if List.is_empty trees || features < 1 || sum_bits < leaf_bits
    then
      raise_s
        [%message
          "BUG: otf_trees needs trees, features and a sum at least as wide as a leaf"
            (features : int)
            (leaf_bits : int)
            (sum_bits : int)];
    List.iter (List.concat_map trees ~f:Tree.splits) ~f:(fun split ->
      if split.feature < 0
         || split.feature >= features
         || not (fits ~bits:feature_bits split.threshold)
      then raise_s [%message "BUG: otf_trees split out of range" (split : Split.t)]);
    List.iter (List.concat_map trees ~f:Tree.leaves) ~f:(fun leaf ->
      if not (fits ~bits:leaf_bits leaf)
      then raise_s [%message "BUG: otf_trees leaf does not fit" (leaf : int)]);
    if not (fits ~bits:sum_bits bias)
    then raise_s [%message "BUG: otf_trees bias does not fit" (bias : int)];
    if stages.select < 0
       || (stages.select > 0 && stages.select >= depth)
       || stages.sum < 0
       || sum_registers > levels
    then
      raise_s
        [%message
          "BUG: otf_trees stages out of range"
            (stages : Stages.t)
            (depth : int)
            (levels : int)]
  ;;

  module I = struct
    type 'a t =
      { clocking : 'a Clocking.t
      ; x : 'a With_valid.t [@bits features * feature_bits]
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t = { y : 'a With_valid.t [@bits if decision then 1 else sum_bits] }
    [@@deriving hardcaml]
  end

  (* Compare, select a leaf per tree, then a binary adder tree, each cut a register. *)
  let create (scope : Scope.t) (i : Signal.t I.t) =
    let reg = Clocking.reg_no_clear i.clocking in
    let x = Array.of_list (split_lsb ~part_width:feature_bits i.x.value) in
    let splits =
      Set.to_list (Set.of_list (module Split) (List.concat_map trees ~f:Tree.splits))
    in
    let%hw_list compares =
      List.map splits ~f:(fun { feature; threshold; equal } ->
        let threshold = of_signed_int ~width:feature_bits threshold in
        reg (if equal then x.(feature) ==: threshold else x.(feature) <=+ threshold))
    in
    (* Each compare, delayed by 0 up to [stages.select] cycles. *)
    let compare =
      Map.of_alist_exn
        (module Split)
        (List.zip_exn
           splits
           (List.map compares ~f:(fun bit ->
              Array.of_list
                (List.folding_map
                   (List.range 0 (stages.select + 1))
                   ~init:bit
                   ~f:(fun bit n ->
                     let bit = if n = 0 then bit else reg bit in
                     bit, bit)))))
    in
    let cut_at cuts n = List.mem cuts n ~equal:Int.equal in
    (* A node's mux runs after the cuts below it, so its select bit waits for them. *)
    let rec select depth : Tree.t -> Signal.t = function
      | Leaf value -> of_signed_int ~width:leaf_bits value
      | Node { split; if_true; if_false } ->
        let late = List.count select_cuts ~f:(fun cut -> cut > depth) in
        let out =
          mux2
            (Map.find_exn compare split).(late)
            (select (depth + 1) if_true)
            (select (depth + 1) if_false)
        in
        if cut_at select_cuts depth then reg out else out
    in
    let%hw_list leaves = List.map trees ~f:(fun tree -> reg (select 0 tree)) in
    let%hw total =
      List.fold
        (List.range 1 (levels + 1))
        ~init:
          (of_signed_int ~width:sum_bits bias
           :: List.map leaves ~f:(sresize ~width:sum_bits))
        ~f:(fun operands level ->
          let operands =
            List.map (List.chunks_of operands ~length:2) ~f:(reduce ~f:( +: ))
          in
          if cut_at sum_cuts level then List.map operands ~f:reg else operands)
      |> List.hd_exn
    in
    let%hw valid = Clocking.pipeline i.clocking ~n:latency i.x.valid in
    { O.y = { valid; value = (if decision then ~:(msb total) else total) } }
  ;;

  let hierarchical ?instance scope i =
    let module H = Hierarchy.In_scope (I) (O) in
    H.hierarchical ?instance ~scope ~name:"otf_trees" create i
  ;;
end
