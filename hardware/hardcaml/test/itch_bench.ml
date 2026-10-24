open! Core
open! Hardcaml

let big_endian value ~bytes =
  String.init bytes ~f:(fun i ->
    Char.of_int_exn ((value lsr (8 * (bytes - 1 - i))) land 0xff))
;;

let header kind ~locate =
  String.of_char kind ^ big_endian locate ~bytes:2 ^ big_endian 0 ~bytes:8
;;

let system_event = header 'S' ~locate:0 ^ "O"

let add ?(locate = 2) ?attribution ~order ~side ~shares ~price () =
  header (if Option.is_some attribution then 'F' else 'A') ~locate
  ^ big_endian order ~bytes:8
  ^ String.of_char side
  ^ big_endian shares ~bytes:4
  ^ "BRVO    "
  ^ big_endian price ~bytes:4
  ^ Option.value attribution ~default:""
;;

let execute ?(locate = 2) ~order ~shares () =
  header 'E' ~locate
  ^ big_endian order ~bytes:8
  ^ big_endian shares ~bytes:4
  ^ big_endian 0 ~bytes:8
;;

let cancel ?(locate = 2) ~order ~shares () =
  header 'X' ~locate ^ big_endian order ~bytes:8 ^ big_endian shares ~bytes:4
;;

let delete ?(locate = 2) ~order () = header 'D' ~locate ^ big_endian order ~bytes:8

let replace ?(locate = 2) ~order ~new_order ~shares ~price () =
  header 'U' ~locate
  ^ big_endian order ~bytes:8
  ^ big_endian new_order ~bytes:8
  ^ big_endian shares ~bytes:4
  ^ big_endian price ~bytes:4
;;

let packet ?count ?(lengths = List.map ~f:String.length) messages =
  "ONNX2FPGA1"
  ^ big_endian 1 ~bytes:8
  ^ big_endian (Option.value count ~default:(List.length messages)) ~bytes:2
  ^ String.concat
      (List.map2_exn (lengths messages) messages ~f:(fun length message ->
         big_endian length ~bytes:2 ^ message))
;;

type beat =
  { data : Bits.t
  ; keep : Bits.t
  ; last : bool
  }

let beats packet =
  let size = String.length packet in
  List.init
    ((size + 7) / 8)
    ~f:(fun index ->
      let chunk =
        String.sub packet ~pos:(8 * index) ~len:(Int.min 8 (size - (8 * index)))
      in
      let bytes =
        List.init 8 ~f:(fun i ->
          Bits.of_unsigned_int
            ~width:8
            (if i < String.length chunk then Char.to_int chunk.[i] else 0))
      in
      { data = Bits.concat_lsb bytes
      ; keep = Bits.of_unsigned_int ~width:8 ((1 lsl String.length chunk) - 1)
      ; last = (8 * index) + 8 >= size
      })
;;

let drive ~data ~keep ~valid ~last beat =
  match beat with
  | None -> valid := Bits.gnd
  | Some beat ->
    data := beat.data;
    keep := beat.keep;
    last := Bits.of_bool beat.last;
    valid := Bits.vdd
;;

let string_of_bits bits ~length =
  String.init length ~f:(fun i ->
    Char.of_int_exn (Bits.to_unsigned_int bits.Bits.:[(8 * i) + 7, 8 * i]))
;;
