open! Core
open! Hardcaml
open! Signal

module Op = struct
  type t =
    | Add
    | Mul
    | Less
    | Less_equal
    | Equal
    | Max
    | Min
    | And
    | Or
    | Where
  [@@deriving sexp_of, enumerate]
end

module Operand = struct
  type t =
    | Port of int
    | Constant of int list
  [@@deriving sexp_of]
end

module Stages = struct
  type t =
    { product : int
    ; rescale : int
    }
  [@@deriving sexp_of]

  let none = { product = 0; rescale = 0 }
end

module type Config = sig
  val lanes : int
  val bits : int
  val op : Op.t
  val operands : Operand.t list
  val zero_points : int list
  val mults : int list
  val tables : int array list
  val mult : int
  val mult_bits : int
  val acc_bits : int
  val shift : int
  val out_zp : int
  val out_min : int
  val out_max : int
  val stages : Stages.t
end

module Make (Config : Config) = struct
  open Config

  let ports =
    List.fold operands ~init:0 ~f:(fun count -> function
      | Port index -> Int.max count (index + 1)
      | Constant _ -> count)
  ;;

  let latency =
    match op with
    | Add | Mul | Less | Less_equal | Equal -> 2 + stages.product + stages.rescale
    | Max | Min | Where -> 1 + stages.rescale
    | And | Or -> 1
  ;;

  let () =
    let arity =
      match op with
      | Where -> 3
      | _ -> 2
    in
    let most_product, most_rescale =
      match op with
      | Add -> 1, 2
      | Mul -> 1, 3
      | Less | Less_equal | Equal -> 1, 0
      | Max | Min | Where -> 0, 1
      | And | Or -> 0, 0
    in
    let constants_allowed =
      match op with
      | Where -> true
      | _ -> false
    in
    if lanes < 1
       || bits < 2
       || List.length operands <> arity
       || List.length zero_points <> arity
       || List.length mults <> arity
       || List.length tables <> arity
       || acc_bits <= bits
    then
      raise_s
        [%message
          "BUG: otf_eltwise operands do not match the operator"
            (op : Op.t)
            (operands : Operand.t list)
            (acc_bits : int)];
    List.iteri (List.zip_exn operands tables) ~f:(fun k (operand, table) ->
      let looked_up =
        match op with
        | Max | Min -> true
        | Where -> k > 0
        | _ -> false
      in
      match operand with
      | Port index when index >= 0 && ((not looked_up) || Array.length table = 1 lsl bits)
        -> ()
      | Constant codes when constants_allowed && List.length codes = lanes -> ()
      | _ ->
        raise_s [%message "BUG: otf_eltwise operand not allowed" (operand : Operand.t)]);
    if stages.product < 0
       || stages.product > most_product
       || stages.rescale < 0
       || stages.rescale > most_rescale
    then raise_s [%message "BUG: otf_eltwise stages out of range" (stages : Stages.t)]
  ;;

  module I = struct
    type 'a t =
      { clocking : 'a Clocking.t
      ; valid : 'a
      ; x : 'a list [@length ports] [@bits lanes * bits]
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t = { y : 'a With_valid.t [@bits lanes * bits] } [@@deriving hardcaml]
  end

  (* Add multiplies by 1 in the rescale, so its first cut registers the accumulator. *)
  module Requant = Otf_requant.Make_pipelined (struct
      let stages =
        match op with
        | Add -> 1 + stages.rescale
        | _ -> stages.rescale
      ;;

      let acc_bits = acc_bits
      let mult_bits = mult_bits
      let shift = shift
      let out_bits = bits
      let out_min = out_min
      let out_max = out_max
      let out_zp = out_zp
    end)

  let lane scope clocking ~ports ~entries k =
    let reg = Clocking.reg_no_clear clocking in
    let cut count signal = if count > 0 then reg signal else signal in
    let wide = bits + 1 in
    let codes =
      List.map operands ~f:(function
        | Port index -> List.nth_exn ports.(index) k
        | Constant codes -> of_signed_int ~width:bits (List.nth_exn codes k))
    in
    let terms =
      List.map2_exn codes zero_points ~f:(fun code zp ->
        sresize code ~width:wide -: of_signed_int ~width:wide zp)
    in
    let truth =
      List.map2_exn codes zero_points ~f:(fun code zp ->
        code <>: of_signed_int ~width:bits zp)
    in
    (* Flipping the sign bit makes the signed code an index from the most negative code. *)
    let sign = of_unsigned_int ~width:bits (1 lsl (bits - 1)) in
    let looked_up index =
      let code = List.nth_exn codes index in
      match List.nth_exn operands index with
      | Port _ -> cut stages.rescale (mux (code ^: sign) entries.(index))
      | Constant _ -> code
    in
    let rescale acc mult =
      (Requant.hierarchical
         scope
         { clocking
         ; acc = sresize acc ~width:acc_bits
         ; mult = of_signed_int ~width:mult_bits mult
         })
        .out
    in
    let weighted () =
      List.map2_exn terms mults ~f:(fun term m ->
        sresize
          (cut stages.product (term *+ of_signed_int ~width:mult_bits m))
          ~width:acc_bits)
      |> reduce ~f:( +: )
    in
    let%hw out =
      match op, terms, truth with
      | Add, _, _ -> rescale (weighted ()) mult
      | Mul, [ a; b ], _ ->
        rescale (reg (sresize (cut stages.product (a *+ b)) ~width:acc_bits)) mult
      | (Less | Less_equal | Equal), _, _ ->
        let acc = reg (weighted ()) in
        let test =
          match op with
          | Less -> msb acc
          | Less_equal -> msb acc |: (acc ==:. 0)
          | _ -> acc ==:. 0
        in
        uresize test ~width:bits
      | Max, _, _ ->
        let a = looked_up 0 in
        let b = looked_up 1 in
        mux2 (a >+ b) a b
      | Min, _, _ ->
        let a = looked_up 0 in
        let b = looked_up 1 in
        mux2 (a <+ b) a b
      | And, _, [ a; b ] -> uresize (a &: b) ~width:bits
      | Or, _, [ a; b ] -> uresize (a |: b) ~width:bits
      | Where, _, condition :: _ ->
        mux2 (cut stages.rescale condition) (looked_up 1) (looked_up 2)
      | _ ->
        raise_s [%message "BUG: otf_eltwise operator and operands disagree" (op : Op.t)]
    in
    reg out
  ;;

  let create (scope : Scope.t) (i : Signal.t I.t) =
    let ports = Array.of_list_map i.x ~f:(fun port -> split_lsb ~part_width:bits port) in
    (* Constants, not a rom: Hardcaml compares memories by size alone, see Sv_int.table. *)
    let entries =
      Array.of_list_map tables ~f:(fun table ->
        List.map (Array.to_list table) ~f:(of_signed_int ~width:bits))
    in
    let%hw_list y =
      List.init lanes ~f:(fun k ->
        lane
          (Scope.sub_scope scope ("lane" ^ Int.to_string k))
          i.clocking
          ~ports
          ~entries
          k)
    in
    let%hw valid = Clocking.pipeline i.clocking ~n:latency i.valid in
    { O.y = { valid; value = concat_lsb y } }
  ;;

  let hierarchical ?instance scope i =
    let module H = Hierarchy.In_scope (I) (O) in
    H.hierarchical ?instance ~scope ~name:"otf_eltwise" create i
  ;;
end
