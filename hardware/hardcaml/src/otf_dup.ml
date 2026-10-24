open! Core
open! Hardcaml
open! Signal

module type Config = sig
  val width : int
  val fanout : int
end

module Make (Config : Config) = struct
  open Config

  let () =
    if fanout < 1 then raise_s [%message "BUG: otf_dup needs a branch" (fanout : int)]
  ;;

  module Beat = Stream.Source (struct
      let width = width
    end)

  module I = struct
    type 'a t =
      { clocking : 'a Clocking.t
      ; s : 'a Beat.t
      ; m : 'a Stream.Dest.t list [@length fanout]
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t =
      { s : 'a Stream.Dest.t
      ; m : 'a Beat.t list [@length fanout]
      }
    [@@deriving hardcaml]
  end

  let create (scope : Scope.t) (i : Signal.t I.t) =
    let spec = Clocking.to_spec i.clocking in
    let%hw sent = wire fanout in
    let m_ready = concat_lsb (List.map i.m ~f:(fun m -> m.ready)) in
    let%hw m_valid = repeat i.s.valid ~count:fanout &: ~:sent in
    let%hw s_ready = reduce ~f:( &: ) (bits_lsb (sent |: m_ready)) in
    sent
    <-- reg
          spec
          ~enable:i.s.valid
          (mux2 s_ready (zero fanout) (sent |: (m_valid &: m_ready)));
    { O.s = { ready = s_ready }
    ; m = List.map (bits_lsb m_valid) ~f:(fun valid -> { Beat.data = i.s.data; valid })
    }
  ;;

  let hierarchical ?instance scope i =
    let module H = Hierarchy.In_scope (I) (O) in
    H.hierarchical ?instance ~scope ~name:"otf_dup" create i
  ;;
end
