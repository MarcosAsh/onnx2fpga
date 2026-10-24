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

  (* One register stage each for the products, the sum and the rescale. *)
  let channel scope clocking ~acts ~weights ~bias ~mult =
    let%hw_list products =
      List.map2_exn
        acts
        (split_lsb ~part_width:wgt_bits (constant weights))
        ~f:(fun a w -> Clocking.reg_no_clear clocking (a *+ w))
    in
    let%hw sum =
      Clocking.reg_no_clear
        clocking
        (constant bias :: List.map products ~f:(sresize ~width:acc_bits)
         |> tree ~arity:2 ~f:(reduce ~f:( +: )))
    in
    let rescaled = Requant.hierarchical scope { acc = sum; mult = constant mult } in
    let%hw out = Clocking.reg_no_clear clocking rescaled.out in
    out
  ;;

  let create (scope : Scope.t) (i : Signal.t I.t) =
    let acts = split_lsb ~part_width:act_bits i.x.value in
    let channels =
      List.mapi
        (List.zip_exn weights (List.zip_exn bias mult))
        ~f:(fun index (weights, (bias, mult)) ->
          let scope = Scope.sub_scope scope ("channel" ^ Int.to_string index) in
          channel scope i.clocking ~acts ~weights ~bias ~mult)
    in
    let%hw valid = Clocking.pipeline i.clocking ~n:latency i.x.valid in
    { O.y = { valid; value = concat_lsb channels } }
  ;;

  let hierarchical ?instance scope i =
    let module H = Hierarchy.In_scope (I) (O) in
    H.hierarchical ?instance ~scope ~name:"otf_dense" create i
  ;;
end
