open! Core
open! Hardcaml
open! Signal

module type Config = sig
  val in_elems : int
  val out_elems : int
  val elem_bits : int
  val total_elems : int
end

module Make (Config : Config) = struct
  open Config

  let () =
    let wide = Int.max in_elems out_elems in
    let narrow = Int.min in_elems out_elems in
    if narrow < 1 || wide % narrow <> 0
    then
      raise_s
        [%message
          "BUG: otf_dwc converts only between integer ratios"
            (in_elems : int)
            (out_elems : int)]
  ;;

  module In = Stream.Source (struct
      let width = in_elems * elem_bits
    end)

  module Out = Stream.Source (struct
      let width = out_elems * elem_bits
    end)

  module I = struct
    type 'a t =
      { clocking : 'a Clocking.t
      ; s : 'a In.t
      ; m : 'a Stream.Dest.t
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t =
      { s : 'a Stream.Dest.t
      ; m : 'a Out.t
      }
    [@@deriving hardcaml]
  end

  (* Collects [up] input beats into one output beat; the first lands lowest. *)
  let pack scope (i : Signal.t I.t) ~up =
    let spec = Clocking.to_spec i.clocking in
    let spec_no_clear = Clocking.to_spec_no_clear i.clocking in
    let%hw full = wire 1 in
    let%hw take = i.s.valid &: ~:full in
    let%hw fill = Sv_int.wrapping_counter spec ~advance:take ~count:up in
    let%hw filled = take &: Sv_int.is_last fill ~count:up in
    full <-- reg spec (mux2 full ~:(i.m.ready) filled);
    let segments =
      List.init up ~f:(fun k ->
        reg spec_no_clear ~enable:(take &: (fill ==:. k) &: ~:(i.clocking.clear)) i.s.data)
    in
    let%hw shreg = concat_lsb segments in
    { O.s = { ready = ~:full }; m = { data = shreg; valid = full } }
  ;;

  (* Holds one input beat and hands it out [down] slices at a time, lowest first. *)
  let unpack scope (i : Signal.t I.t) ~down =
    let spec = Clocking.to_spec i.clocking in
    let spec_no_clear = Clocking.to_spec_no_clear i.clocking in
    let%hw loaded = wire 1 in
    let%hw load = ~:loaded &: i.s.valid in
    let%hw step = loaded &: i.m.ready in
    let%hw slot = Sv_int.wrapping_counter spec ~advance:step ~count:down in
    let%hw drained = step &: Sv_int.is_last slot ~count:down in
    loaded <-- reg spec (mux2 loaded ~:drained load);
    let%hw holding = reg spec_no_clear ~enable:(load &: ~:(i.clocking.clear)) i.s.data in
    let%hw slice = mux slot (split_lsb ~part_width:(out_elems * elem_bits) holding) in
    { O.s = { ready = ~:loaded }; m = { data = slice; valid = loaded } }
  ;;

  let create (scope : Scope.t) (i : Signal.t I.t) =
    if in_elems = out_elems
    then { O.s = { ready = i.m.ready }; m = { data = i.s.data; valid = i.s.valid } }
    else if out_elems > in_elems
    then pack scope i ~up:(out_elems / in_elems)
    else unpack scope i ~down:(in_elems / out_elems)
  ;;

  let hierarchical ?instance scope i =
    let module H = Hierarchy.In_scope (I) (O) in
    H.hierarchical ?instance ~scope ~name:"otf_dwc" create i
  ;;
end
