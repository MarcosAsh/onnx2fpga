open! Core
open! Hardcaml
open! Signal

module type Config = sig
  val mw : int
  val mh : int
  val act_bits : int
  val wgt_bits : int
  val acc_bits : int
  val out_bits : int
  val mult_bits : int
  val shift : int
  val out_min : int
  val out_max : int
  val out_zp : int
  val arithmetic : [ `Multiplier | `Constant ]
  val weights : Bits.t list
  val bias : Bits.t list
  val mult : Bits.t list
end

module Make (Config : Config) = struct
  open Config

  let latency = 3

  let () =
    let check name words ~width =
      if List.length words <> mh
         || List.exists words ~f:(fun word -> Bits.width word <> width)
      then
        raise_s
          [%message
            "BUG: otf_dense table has the wrong shape" name (mh : int) (width : int)]
    in
    if mw < 1 || mh < 1
    then
      raise_s [%message "BUG: otf_dense needs a non empty matrix" (mw : int) (mh : int)];
    if acc_bits < act_bits + wgt_bits
    then
      raise_s
        [%message
          "BUG: otf_dense accumulator is narrower than one product"
            (acc_bits : int)
            (act_bits : int)
            (wgt_bits : int)];
    check "weights" weights ~width:(mw * wgt_bits);
    check "bias" bias ~width:acc_bits;
    check "mult" mult ~width:mult_bits
  ;;

  module I = struct
    type 'a t =
      { clocking : 'a Clocking.t
      ; x : 'a With_valid.t [@bits mw * act_bits]
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t = { y : 'a With_valid.t [@bits mh * out_bits] } [@@deriving hardcaml]
  end

  module Requant = Otf_requant.Make (struct
      let acc_bits = acc_bits
      let mult_bits = mult_bits
      let shift = shift
      let out_bits = out_bits
      let out_min = out_min
      let out_max = out_max
      let out_zp = out_zp
    end)

  let constant word = of_constant (Bits.to_constant word)

  (* Reitwiesner's non adjacent form as (positive, position), most significant first. *)
  let csd value =
    let rec digits value position acc =
      if value = 0
      then acc
      else if value land 1 = 0
      then digits (value asr 1) (position + 1) acc
      else (
        let digit = 2 - (value land 3) in
        digits ((value - digit) asr 1) (position + 1) ((digit > 0, position) :: acc))
    in
    digits value 0 []
  ;;

  (* Sums (positive, value) terms with adders and subtractors, keeping the first sign. *)
  let signed_sum terms =
    tree
      terms
      ~arity:2
      ~f:
        (reduce ~f:(fun (positive, a) (same, b) ->
           positive, if Bool.equal positive same then a +: b else a -: b))
  ;;

  let rec odd_part magnitude shift =
    if magnitude land 1 = 0
    then odd_part (magnitude asr 1) (shift + 1)
    else magnitude, shift
  ;;

  let weight_values =
    List.map weights ~f:(fun word ->
      List.map (Bits.split_lsb ~part_width:wgt_bits word) ~f:Bits.to_signed_int)
  ;;

  (* Input [k] times each odd magnitude its column needs, from shared 2^j +- 1 pairs. *)
  let products scope clocking k a =
    let a = sresize a ~width:(act_bits + wgt_bits) in
    let shifted = Hashtbl.create (module Int) in
    let times c =
      Hashtbl.find_or_add shifted c ~default:(fun () ->
        snd (signed_sum (List.map (csd c) ~f:(fun (positive, by) -> positive, sll a ~by))))
    in
    let rec pairs = function
      | (positive, high) :: (same, low) :: digits ->
        let c = (1 lsl (high - low)) + if Bool.equal positive same then 1 else -1 in
        (positive, sll (times c) ~by:low) :: pairs digits
      | [ (positive, by) ] -> [ positive, sll a ~by ]
      | [] -> []
    in
    List.filter_map weight_values ~f:(fun row ->
      match List.nth_exn row k with
      | 0 -> None
      | w -> Some (fst (odd_part (abs w) 0)))
    |> List.dedup_and_sort ~compare:Int.compare
    |> List.map ~f:(fun m ->
      let product = snd (signed_sum (pairs (csd m))) in
      ( m
      , Scope.naming
          scope
          (Clocking.reg_no_clear
             clocking
             (sresize product ~width:(act_bits + Int.ceil_log2 m)))
          (sprintf "x%d_times%d" k m) ))
  ;;

  (* One term per nonzero weight, registered: the first of the three stages. *)
  let terms scope clocking ~acts ~shared row =
    List.filter_mapi row ~f:(fun k w ->
      let a = List.nth_exn acts k in
      match arithmetic with
      | _ when w = 0 -> None
      | `Multiplier ->
        let product = a *+ of_signed_int ~width:(wgt_bits + 1) (abs w) in
        let product =
          Scope.naming
            scope
            (Clocking.reg_no_clear
               clocking
               (sresize product ~width:(act_bits + wgt_bits)))
            (sprintf "x%d_product" k)
        in
        Some (w > 0, sresize product ~width:acc_bits)
      | `Constant ->
        let m, by = odd_part (abs w) 0 in
        let product = List.Assoc.find_exn (List.nth_exn shared k) m ~equal:Int.equal in
        Some (w > 0, sll (sresize product ~width:acc_bits) ~by))
  ;;

  let channel scope clocking ~acts ~shared ~row ~bias ~mult =
    let%hw sum =
      Clocking.reg_no_clear
        clocking
        (snd
           (signed_sum ((true, constant bias) :: terms scope clocking ~acts ~shared row)))
    in
    let rescaled = Requant.hierarchical scope { acc = sum; mult = constant mult } in
    let%hw out = Clocking.reg_no_clear clocking rescaled.out in
    out
  ;;

  let create (scope : Scope.t) (i : Signal.t I.t) =
    let acts = split_lsb ~part_width:act_bits i.x.value in
    let shared =
      match arithmetic with
      | `Multiplier -> []
      | `Constant -> List.mapi acts ~f:(products scope i.clocking)
    in
    let channels =
      List.mapi
        (List.zip_exn weight_values (List.zip_exn bias mult))
        ~f:(fun index (row, (bias, mult)) ->
          let scope = Scope.sub_scope scope ("channel" ^ Int.to_string index) in
          channel scope i.clocking ~acts ~shared ~row ~bias ~mult)
    in
    let%hw valid = Clocking.pipeline i.clocking ~n:latency i.x.valid in
    { O.y = { valid; value = concat_lsb channels } }
  ;;

  let hierarchical ?instance scope i =
    let module H = Hierarchy.In_scope (I) (O) in
    H.hierarchical ?instance ~scope ~name:"otf_dense" create i
  ;;
end

module type Reloadable_config = sig
  include Config

  val base : int
end

module Make_reloadable (Config : Reloadable_config) = struct
  open Config
  module Frozen = Make (Config)

  let latency = Frozen.latency
  let words = (mh * mw) + mh + mh

  let () =
    match arithmetic with
    | `Constant ->
      raise_s [%message "BUG: otf_dense reloadable weights need general multipliers"]
    | `Multiplier ->
      if Int.max acc_bits mult_bits > 32
      then
        raise_s
          [%message
            "BUG: otf_dense bank fields must fit a 32 bit word"
              (acc_bits : int)
              (mult_bits : int)]
  ;;

  (* The bank order: weights channel by channel, then the biases, then the multipliers. *)
  let fields =
    List.concat_map Frozen.weight_values ~f:(fun row ->
      List.map row ~f:(Bits.of_signed_int ~width:wgt_bits))
    @ bias
    @ mult
  ;;

  let image = List.map fields ~f:(Bits.uresize ~width:32)

  module I = struct
    type 'a t =
      { clocking : 'a Clocking.t
      ; x : 'a With_valid.t [@bits mw * act_bits]
      ; x_bank : 'a
      ; store : 'a Weight_banks.Write.t [@rtlprefix "store_"]
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t =
      { y : 'a With_valid.t [@bits mh * out_bits]
      ; y_bank : 'a
      }
    [@@deriving hardcaml]
  end

  let create (scope : Scope.t) (i : Signal.t I.t) =
    let spec = Clocking.to_spec_no_clear i.clocking in
    let reg = Clocking.reg_no_clear i.clocking in
    let banks =
      Array.of_list_mapi fields ~f:(fun index initial ->
        let bank number =
          let enable =
            i.store.write
            &: (i.store.bank ==:. number)
            &: (i.store.address ==:. base + index)
          in
          Scope.naming
            scope
            (Signal.reg
               spec
               ~initialize_to:initial
               ~enable
               (sel_bottom i.store.data ~width:(Bits.width initial)))
            (sprintf "word%d_bank%d" index number)
        in
        bank 0, bank 1)
    in
    let pick index tag =
      let zero, one = banks.(index) in
      mux2 tag one zero
    in
    let%hw sum_bank = reg i.x_bank in
    let%hw rescale_bank = reg sum_bank in
    let%hw y_bank = reg rescale_bank in
    let acts = split_lsb ~part_width:act_bits i.x.value in
    let channel c =
      let scope = Scope.sub_scope scope ("channel" ^ Int.to_string c) in
      let products =
        List.mapi acts ~f:(fun k a ->
          Scope.naming
            scope
            (reg (a *+ pick ((c * mw) + k) i.x_bank))
            (sprintf "x%d_product" k))
      in
      let%hw sum =
        reg
          (tree
             ~arity:2
             ~f:(reduce ~f:( +: ))
             (pick ((mh * mw) + c) sum_bank
              :: List.map products ~f:(sresize ~width:acc_bits)))
      in
      let rescaled =
        Frozen.Requant.hierarchical
          scope
          { acc = sum; mult = pick ((mh * mw) + mh + c) rescale_bank }
      in
      let%hw out = reg rescaled.out in
      out
    in
    let%hw valid = Clocking.pipeline i.clocking ~n:latency i.x.valid in
    { O.y = { valid; value = concat_lsb (List.init mh ~f:channel) }; y_bank }
  ;;

  let hierarchical ?instance scope i =
    let module H = Hierarchy.In_scope (I) (O) in
    H.hierarchical ?instance ~scope ~name:"otf_dense" create i
  ;;
end
