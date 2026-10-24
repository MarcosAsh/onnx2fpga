open! Core
open! Hardcaml
open! Signal

module type Config = sig
  val width : int
  val depth : int
end

module Make (Config : Config) = struct
  open Config

  let () =
    if depth < 1 then raise_s [%message "BUG: otf_fifo needs a depth" (depth : int)]
  ;;

  module Beat = Stream.Source (struct
      let width = width
    end)

  module I = struct
    type 'a t =
      { clocking : 'a Clocking.t
      ; s : 'a Beat.t
      ; m : 'a Stream.Dest.t
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t =
      { s : 'a Stream.Dest.t
      ; m : 'a Beat.t
      }
    [@@deriving hardcaml]
  end

  let ptr_bits = address_bits_for depth
  let cnt_bits = num_bits_to_represent depth

  let create (scope : Scope.t) (i : Signal.t I.t) =
    let spec = Clocking.to_spec i.clocking in
    let%hw count = wire cnt_bits in
    let%hw s_ready = count <>:. depth in
    let%hw m_valid = count <>:. 0 in
    let%hw push = i.s.valid &: s_ready in
    let%hw pop = m_valid &: i.m.ready in
    let advance ptr =
      if depth = 1 then zero ptr_bits else mod_counter ~max:(depth - 1) ptr
    in
    let%hw wptr = reg_fb spec ~enable:push ~width:ptr_bits ~f:advance in
    let%hw rptr = reg_fb spec ~enable:pop ~width:ptr_bits ~f:advance in
    count
    <-- reg
          spec
          (mux2 (push &: ~:pop) (count +:. 1) @@ mux2 (pop &: ~:push) (count -:. 1) count);
    let stored =
      multiport_memory
        ~name:"stored"
        depth
        ~write_ports:
          [| { write_clock = i.clocking.clock
             ; write_address = wptr
             ; write_enable = push &: ~:(i.clocking.clear)
             ; write_data = i.s.data
             }
          |]
        ~read_addresses:[| rptr |]
    in
    let%hw m_data = stored.(0) in
    { O.s = { ready = s_ready }; m = { data = m_data; valid = m_valid } }
  ;;

  let hierarchical ?instance scope i =
    let module H = Hierarchy.In_scope (I) (O) in
    H.hierarchical ?instance ~scope ~name:"otf_fifo" create i
  ;;
end
