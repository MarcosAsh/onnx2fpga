open! Core

let character ?(stop = true) ~clocks_per_bit byte =
  let bits = (false :: List.init 8 ~f:(fun k -> (byte lsr k) land 1 = 1)) @ [ stop ] in
  let edge k = Float.iround_nearest_exn (Float.of_int k *. clocks_per_bit) in
  List.concat_mapi bits ~f:(fun k level ->
    List.init (edge (k + 1) - edge k) ~f:(fun _ -> level))
;;

let line ~clocks_per_bit ?(gap = fun () -> 0) bytes =
  List.concat_map bytes ~f:(fun byte ->
    character ~clocks_per_bit byte
    @ List.init
        (Float.iround_nearest_exn (Float.of_int (gap ()) *. clocks_per_bit))
        ~f:(fun _ -> true))
;;

let idle cycles = List.init cycles ~f:(fun _ -> true)
let break_ cycles = List.init cycles ~f:(fun _ -> false)

module Receiver = struct
  type t =
    { clocks_per_bit : float
    ; mutable cycle : int
    ; mutable last : bool
    ; mutable start : int option
    ; mutable index : int
    ; mutable shift : int
    ; mutable received : int list
    ; mutable framing_errors : int
    }

  let create ~clocks_per_bit =
    { clocks_per_bit
    ; cycle = 0
    ; last = true
    ; start = None
    ; index = 0
    ; shift = 0
    ; received = []
    ; framing_errors = 0
    }
  ;;

  let step t level =
    (match t.start with
     | None ->
       if t.last && not level
       then (
         t.start <- Some t.cycle;
         t.index <- 0;
         t.shift <- 0)
     | Some start ->
       let sample =
         Float.of_int start +. (t.clocks_per_bit *. (Float.of_int t.index +. 0.5))
       in
       if Float.(Float.of_int t.cycle >= sample)
       then (
         match t.index with
         | 0 when level -> t.start <- None
         | 9 ->
           t.start <- None;
           if level
           then t.received <- t.shift :: t.received
           else t.framing_errors <- t.framing_errors + 1
         | index ->
           if index >= 1 && level then t.shift <- t.shift lor (1 lsl (index - 1));
           t.index <- index + 1));
    t.last <- level;
    t.cycle <- t.cycle + 1
  ;;

  let received t = List.rev t.received
  let framing_errors t = t.framing_errors
end
