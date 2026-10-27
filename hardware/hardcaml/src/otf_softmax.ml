open! Core
open! Hardcaml
open! Signal

module Mode = struct
  type t =
    | Divide
    | Log
    | Root
    | Max
  [@@deriving sexp_of]
end

module Stages = struct
  type t =
    { max : int
    ; table : int
    ; sum : int
    ; normalize : int
    ; rescale : int
    }
  [@@deriving sexp_of]

  let none = { max = 0; table = 0; sum = 0; normalize = 0; rescale = 0 }
end

module type Config = sig
  val lanes : int
  val in_bits : int
  val subtract_max : bool
  val mode : Mode.t
  val sum_table : int array
  val sum_bits : int
  val numer_table : int array
  val numer_bits : int
  val total_bits : int
  val min_exp : int
  val mantissa_bits : int
  val scale_table : int array
  val exp_table : int array
  val scale_bits : int
  val shift : int
  val out_bits : int
  val out_zp : int
  val out_min : int
  val out_max : int
  val stages : Stages.t
end

module Make (Config : Config) = struct
  open Config

  let levels = Int.ceil_log2 lanes
  let exps = total_bits - min_exp
  let exp_bits = Int.max 1 (Int.ceil_log2 exps)
  let shared_numer = Array.is_empty numer_table

  let latency =
    match mode with
    | Max -> 1 + stages.max + 1 + stages.table + 1 + stages.rescale
    | Divide | Log | Root ->
      (if subtract_max then 1 + stages.max else 0)
      + 1
      + stages.table
      + 1
      + stages.sum
      + 1
      + stages.normalize
      + 1
      + stages.rescale
  ;;

  (* Root shifts by half of the leading one's position, the parity going to the table. *)
  let root_shifts = (exps - 1) / 2

  (* Divide shifts each numerator left by up to [exps - 1]; Log subtracts the logarithm. *)
  let acc_bits =
    match mode with
    | Divide -> numer_bits + exps - 1
    | Root -> numer_bits + root_shifts
    | Log -> Int.max numer_bits scale_bits + 1
    | Max -> numer_bits
  ;;

  let () =
    let fits ~bits ~signed value =
      if signed
      then value >= -(1 lsl (bits - 1)) && value < 1 lsl (bits - 1)
      else value >= 0 && value < 1 lsl bits
    in
    let size = 1 lsl in_bits in
    let summed = not (Poly.equal mode Mode.Max) in
    let scales =
      match mode with
      | Divide | Log -> 1 lsl mantissa_bits
      | Root -> 1 lsl (mantissa_bits + 1)
      | Max -> size
    in
    if lanes < 1
       || in_bits < 1
       || exps < 1
       || min_exp < 0
       || mantissa_bits < 1
       || (summed && Array.length sum_table <> size)
       || (not (Array.for_all sum_table ~f:(fits ~bits:sum_bits ~signed:false)))
       || (shared_numer && not summed)
       || (shared_numer && numer_bits <> sum_bits + 1)
       || ((not shared_numer)
           && (Array.length numer_table <> size
               || not (Array.for_all numer_table ~f:(fits ~bits:numer_bits ~signed:true))
              ))
       || Array.length scale_table <> scales
       || (not (Array.for_all scale_table ~f:(fits ~bits:scale_bits ~signed:true)))
       || lanes * Array.fold sum_table ~init:0 ~f:Int.max >= 1 lsl total_bits
    then
      raise_s
        [%message
          "BUG: otf_softmax tables do not match their widths"
            (lanes : int)
            (in_bits : int)
            (total_bits : int)
            (min_exp : int)];
    (match mode with
     | Divide -> ()
     | Root ->
       if subtract_max || min_exp % 2 <> 0
       then raise_s [%message "BUG: otf_softmax root needs no max and an even min_exp"]
     | Max ->
       if subtract_max || stages.sum <> 0 || stages.normalize <> 0
       then raise_s [%message "BUG: otf_softmax max has no sum and no normalisation"]
     | Log ->
       if (not subtract_max)
          || Array.length exp_table <> exps
          || not (Array.for_all exp_table ~f:(fits ~bits:scale_bits ~signed:true))
       then raise_s [%message "BUG: otf_softmax log needs the max and an exponent table"]);
    let tree_ok stages levels = stages >= 0 && (stages = 0 || stages < levels) in
    let has_max = subtract_max || Poly.equal mode Mode.Max in
    if (not (tree_ok stages.max levels))
       || (not (tree_ok stages.sum levels))
       || ((not has_max) && stages.max > 0)
       || stages.table < 0
       || stages.table > 1
       || stages.normalize < 0
       || stages.normalize > 2
       || (stages.normalize = 2 && exps < 3)
       || stages.rescale < 0
       || stages.rescale > 3
    then raise_s [%message "BUG: otf_softmax stages out of range" (stages : Stages.t)]
  ;;

  module I = struct
    type 'a t =
      { clocking : 'a Clocking.t
      ; x : 'a With_valid.t [@bits lanes * in_bits]
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t = { y : 'a With_valid.t [@bits lanes * out_bits] } [@@deriving hardcaml]
  end

  module Requant = Otf_requant.Make_pipelined (struct
      let stages = stages.rescale
      let acc_bits = acc_bits

      let mult_bits =
        match mode with
        | Divide | Root | Max -> scale_bits
        | Log -> 2
      ;;

      let shift = shift
      let out_bits = out_bits
      let out_min = out_min
      let out_max = out_max
      let out_zp = out_zp
    end)

  let constants table ~width ~signed =
    List.map (Array.to_list table) ~f:(fun value ->
      if signed then of_signed_int ~width value else of_unsigned_int ~width value)
  ;;

  let pick select = function
    | [ only ] -> only
    | items -> mux select items
  ;;

  (* The [mantissa_bits] bits below bit [k] of [s], zeros filling in below bit 0. *)
  let mantissa s k =
    if k >= mantissa_bits
    then s.:[k - 1, k - mantissa_bits]
    else if k = 0
    then zero mantissa_bits
    else s.:[k - 1, 0] @: zero (mantissa_bits - k)
  ;;

  let largest clocking codes =
    Otf_reduce.tree
      clocking
      ~cuts:(Otf_reduce.cuts ~levels ~stages:stages.max)
      ~f:(fun a b -> mux2 (a >+ b) a b)
      codes
  ;;

  (* A table index: the code counted from the most negative. *)
  let from_bottom code = code ^: of_unsigned_int ~width:in_bits (1 lsl (in_bits - 1))

  (* Max: every numerator times the scale table at the largest code; nothing is summed. *)
  let create_max scope (i : Signal.t I.t) =
    let reg = Clocking.reg_no_clear i.clocking in
    let delay n = Clocking.pipeline_no_clear i.clocking ~n in
    let codes = split_lsb ~part_width:in_bits i.x.value in
    let%hw top = reg (largest i.clocking codes) in
    let index = if stages.table > 0 then reg (from_bottom top) else from_bottom top in
    let%hw scale =
      reg (mux index (constants scale_table ~width:scale_bits ~signed:true))
    in
    let numers = constants numer_table ~width:numer_bits ~signed:true in
    let%hw_list y =
      List.map codes ~f:(fun code ->
        let code = delay (2 + stages.max + stages.table) code in
        let acc = sresize (mux (from_bottom code) numers) ~width:acc_bits in
        reg (Requant.hierarchical scope { clocking = i.clocking; acc; mult = scale }).out)
    in
    let%hw valid = Clocking.pipeline i.clocking ~n:latency i.x.valid in
    { O.y = { valid; value = concat_lsb y } }
  ;;

  let create_summed scope (i : Signal.t I.t) =
    let reg = Clocking.reg_no_clear i.clocking in
    let delay n = Clocking.pipeline_no_clear i.clocking ~n in
    let cut n signal = if n > 0 then reg signal else signal in
    let codes = split_lsb ~part_width:in_bits i.x.value in
    (* Lane tables read at [max - x], or at the code counted from the most negative. *)
    let indices =
      if subtract_max
      then (
        let%hw top = reg (largest i.clocking codes) in
        List.map codes ~f:(fun code ->
          let wide = in_bits + 1 in
          sel_bottom
            (sresize top ~width:wide -: sresize (delay (1 + stages.max) code) ~width:wide)
            ~width:in_bits))
      else List.map codes ~f:from_bottom
    in
    let sums = constants sum_table ~width:sum_bits ~signed:false in
    let numers = constants numer_table ~width:numer_bits ~signed:true in
    let looked_up =
      List.map indices ~f:(fun index ->
        let index = cut stages.table index in
        let a = reg (mux index sums) in
        let b =
          if shared_numer then uresize a ~width:numer_bits else reg (mux index numers)
        in
        a, b)
    in
    let%hw total =
      reg
        (Otf_reduce.tree
           i.clocking
           ~cuts:(Otf_reduce.cuts ~levels ~stages:stages.sum)
           ~f:( +: )
           (List.map looked_up ~f:(fun (a, _) -> uresize a ~width:total_bits)))
    in
    let numers = List.map looked_up ~f:(fun (_, b) -> delay (1 + stages.sum) b) in
    (* The leading one of [total] from [min_exp] up, as [k - min_exp], and its mantissa. *)
    let found =
      List.rev_map
        (List.range (min_exp + 1) total_bits)
        ~f:(fun k ->
          { With_valid.valid = total.:(k)
          ; value = of_unsigned_int ~width:exp_bits (k - min_exp) @: mantissa total k
          })
    in
    let default = zero exp_bits @: mantissa total min_exp in
    (* A second cut splits the search: each half registered, then the high half if it has
       a one. *)
    let%hw normalized =
      match stages.normalize with
      | 2 ->
        let #(high, low) = List.split_n found (List.length found / 2) in
        let any_high = reg (reduce (List.map high ~f:(fun f -> f.valid)) ~f:( |: )) in
        let high = reg (priority_select_with_default high ~default) in
        let low = reg (priority_select_with_default low ~default) in
        reg (mux2 any_high high low)
      | n -> cut n (priority_select_with_default found ~default)
    in
    let%hw is_zero = delay stages.normalize (total ==:. 0) in
    let numers = List.map numers ~f:(delay stages.normalize) in
    let exp = sel_top normalized ~width:exp_bits in
    (* Root reads its table at the exponent's parity above the mantissa. *)
    let looked =
      mux
        (sel_bottom
           normalized
           ~width:(if Poly.equal mode Mode.Root then mantissa_bits + 1 else mantissa_bits))
        (constants scale_table ~width:scale_bits ~signed:true)
    in
    let%hw scale =
      reg
        (match mode with
         | Divide | Root | Max -> mux2 is_zero (zero scale_bits) looked
         | Log -> looked +: pick exp (constants exp_table ~width:scale_bits ~signed:true))
    in
    let accs =
      List.map numers ~f:(fun b ->
        let b = sresize b ~width:acc_bits in
        match mode with
        | Divide -> reg (pick exp (List.init exps ~f:(fun j -> sll b ~by:(exps - 1 - j))))
        | Root when root_shifts = 0 -> reg b
        | Root ->
          reg
            (pick
               (drop_bottom exp ~width:1)
               (List.init (root_shifts + 1) ~f:(fun j -> sll b ~by:(root_shifts - j))))
        | Log | Max -> reg b)
    in
    let%hw_list y =
      List.map accs ~f:(fun acc ->
        let acc, mult =
          match mode with
          | Divide | Root | Max -> acc, scale
          | Log -> acc -: sresize scale ~width:acc_bits, of_signed_int ~width:2 1
        in
        reg (Requant.hierarchical scope { clocking = i.clocking; acc; mult }).out)
    in
    let%hw valid = Clocking.pipeline i.clocking ~n:latency i.x.valid in
    { O.y = { valid; value = concat_lsb y } }
  ;;

  let create scope i =
    match mode with
    | Max -> create_max scope i
    | Divide | Log | Root -> create_summed scope i
  ;;

  let hierarchical ?instance scope i =
    let module H = Hierarchy.In_scope (I) (O) in
    H.hierarchical ?instance ~scope ~name:"otf_softmax" create i
  ;;
end
