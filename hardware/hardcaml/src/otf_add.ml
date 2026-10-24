open! Core
open! Hardcaml
open! Signal

module type Config = sig
  val pe : int
  val in_bits : int
  val out_bits : int
  val acc_bits : int
  val left_mult : int
  val right_mult : int
  val shift : int
  val out_min : int
  val out_max : int
  val bias : int
  val out_zp : int
end

module Make (Config : Config) = struct
  open Config

  module In = Stream.Source (struct
      let width = pe * in_bits
    end)

  module Out = Stream.Source (struct
      let width = pe * out_bits
    end)

  module I = struct
    type 'a t =
      { clocking : 'a Clocking.t
      ; a : 'a In.t
      ; b : 'a In.t
      ; m : 'a Stream.Dest.t
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t =
      { a : 'a Stream.Dest.t
      ; b : 'a Stream.Dest.t
      ; m : 'a Out.t
      }
    [@@deriving hardcaml]
  end

  module Requant = Otf_requant.Make (struct
      let acc_bits = acc_bits
      let mult_bits = 2
      let shift = shift
      let out_bits = out_bits
      let out_min = out_min
      let out_max = out_max
      let out_zp = out_zp
    end)

  let scaled lane factor =
    sel_bottom
      (sresize lane ~width:acc_bits *+ Sv_int.cast ~width:acc_bits factor)
      ~width:acc_bits
  ;;

  let create (scope : Scope.t) (i : Signal.t I.t) =
    let%hw both = i.a.valid &: i.b.valid in
    let%hw taken = both &: i.m.ready in
    let data =
      List.map2_exn
        (split_lsb ~part_width:in_bits i.a.data)
        (split_lsb ~part_width:in_bits i.b.data)
        ~f:(fun left right ->
          let acc =
            scaled left left_mult
            +: scaled right right_mult
            +: Sv_int.cast ~width:acc_bits bias
          in
          (Requant.hierarchical scope { acc; mult = of_signed_int ~width:2 1 }).out)
    in
    { O.a = { ready = taken }
    ; b = { ready = taken }
    ; m = { data = concat_lsb data; valid = both }
    }
  ;;

  let hierarchical ?instance scope i =
    let module H = Hierarchy.In_scope (I) (O) in
    H.hierarchical ?instance ~scope ~name:"otf_add" create i
  ;;
end
