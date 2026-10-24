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

  let rec eval t x =
    match t with
    | Leaf value -> value
    | Node { split = { feature; threshold; equal }; if_true; if_false } ->
      let taken = if equal then x.(feature) = threshold else x.(feature) <= threshold in
      eval (if taken then if_true else if_false) x
  ;;
end

module type Config = sig
  val features : int
  val feature_bits : int
  val leaf_bits : int
  val sum_bits : int
  val trees : Tree.t list
  val bias : int
  val decision : bool
end

let arity = 4

let rec adder_stages operands =
  if operands <= 1 then 0 else 1 + adder_stages ((operands + arity - 1) / arity)
;;

module Make (Config : Config) = struct
  open Config

  let latency = 2 + adder_stages (List.length trees + 1)

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
    then raise_s [%message "BUG: otf_trees bias does not fit" (bias : int)]
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

  (* Compare, select a leaf per tree, then adder stages of fan in [arity]. *)
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
    let compare = Map.of_alist_exn (module Split) (List.zip_exn splits compares) in
    let rec select : Tree.t -> Signal.t = function
      | Leaf value -> of_signed_int ~width:leaf_bits value
      | Node { split; if_true; if_false } ->
        mux2 (Map.find_exn compare split) (select if_true) (select if_false)
    in
    let%hw_list leaves = List.map trees ~f:(fun tree -> reg (select tree)) in
    let rec add = function
      | [ total ] -> total
      | operands ->
        let%hw_list partial =
          List.map (List.chunks_of operands ~length:arity) ~f:(fun group ->
            reg (reduce group ~f:( +: )))
        in
        add partial
    in
    let%hw total =
      add
        (of_signed_int ~width:sum_bits bias
         :: List.map leaves ~f:(sresize ~width:sum_bits))
    in
    let%hw valid = Clocking.pipeline i.clocking ~n:latency i.x.valid in
    { O.y = { valid; value = (if decision then ~:(msb total) else total) } }
  ;;

  let hierarchical ?instance scope i =
    let module H = Hierarchy.In_scope (I) (O) in
    H.hierarchical ?instance ~scope ~name:"otf_trees" create i
  ;;
end
