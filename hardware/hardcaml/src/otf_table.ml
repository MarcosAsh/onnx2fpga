open! Core
open! Hardcaml
open! Signal

module type Config = sig
  val lanes : int
  val in_bits : int
  val out_bits : int
  val tables : int array list
  val lane_table : int list
end

module Make (Config : Config) = struct
  open Config

  let latency = 1

  let () =
    let fits value = value >= -(1 lsl (out_bits - 1)) && value < 1 lsl (out_bits - 1) in
    if lanes < 1 || in_bits < 1 || in_bits > 12 || out_bits < 1 || List.is_empty tables
    then
      raise_s
        [%message
          "BUG: otf_table needs lanes, a table and inputs of 1 to 12 bits"
            (lanes : int)
            (in_bits : int)
            (out_bits : int)];
    List.iteri tables ~f:(fun index table ->
      if Array.length table <> 1 lsl in_bits || not (Array.for_all table ~f:fits)
      then raise_s [%message "BUG: otf_table table has the wrong shape" (index : int)]);
    if List.length lane_table <> lanes
       || List.exists lane_table ~f:(fun t -> t < 0 || t >= List.length tables)
    then
      raise_s
        [%message "BUG: otf_table lanes pick missing tables" (lane_table : int list)]
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

  (* Constants, not a rom: Hardcaml compares memories by size alone, see Sv_int.table. *)
  let create (scope : Scope.t) (i : Signal.t I.t) =
    let entries =
      Array.of_list_map tables ~f:(fun table ->
        List.map (Array.to_list table) ~f:(of_signed_int ~width:out_bits))
    in
    (* Flipping the sign bit makes the signed code an index from the most negative code. *)
    let sign = of_unsigned_int ~width:in_bits (1 lsl (in_bits - 1)) in
    let%hw_list y =
      List.map2_exn
        (split_lsb ~part_width:in_bits i.x.value)
        lane_table
        ~f:(fun code table ->
          Clocking.reg_no_clear i.clocking (mux (code ^: sign) entries.(table)))
    in
    let%hw valid = Clocking.pipeline i.clocking ~n:latency i.x.valid in
    { O.y = { valid; value = concat_lsb y } }
  ;;

  let hierarchical ?instance scope i =
    let module H = Hierarchy.In_scope (I) (O) in
    H.hierarchical ?instance ~scope ~name:"otf_table" create i
  ;;
end
