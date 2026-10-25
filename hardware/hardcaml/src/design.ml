open! Core
open! Hardcaml

module Json = struct
  let fail what json = raise_s [%message "design.json" what (json : Jsonaf.t)]

  let field json name =
    match json with
    | `Object fields ->
      (match List.Assoc.find fields name ~equal:String.equal with
       | Some value -> value
       | None -> fail ("missing field " ^ name) json)
    | _ -> fail "expected an object" json
  ;;

  let int json =
    match json with
    | `Number text ->
      (match Int.of_string_opt text with
       | Some value -> value
       | None -> fail "expected an integer" json)
    | _ -> fail "expected an integer" json
  ;;

  let string json =
    match json with
    | `String text -> text
    | _ -> fail "expected a string" json
  ;;

  let list json ~f =
    match json with
    | `Array items -> List.map items ~f
    | _ -> fail "expected an array" json
  ;;

  let find json name =
    match json with
    | `Object fields -> List.Assoc.find fields name ~equal:String.equal
    | _ -> fail "expected an object" json
  ;;

  let assoc json ~f =
    match json with
    | `Object fields -> List.map fields ~f:(fun (name, value) -> name, f value)
    | _ -> fail "expected an object" json
  ;;
end

module Memory = struct
  type t =
    { elem_bits : int
    ; lanes : int
    ; values : int list
    }
  [@@deriving sexp_of]

  let of_json json =
    { elem_bits = Json.int (Json.field json "elem_bits")
    ; lanes = Json.int (Json.field json "lanes")
    ; values = Json.list (Json.field json "values") ~f:Json.int
    }
  ;;

  let words { elem_bits; lanes; values } =
    if lanes < 1 || List.length values % lanes <> 0
    then raise_s [%message "design.json: memory does not divide into words" (lanes : int)];
    List.chunks_of values ~length:lanes
    |> List.map ~f:(fun lanes ->
      Bits.concat_lsb (List.map lanes ~f:(Bits.of_unsigned_int ~width:elem_bits)))
  ;;
end

module Node = struct
  type t =
    { name : string
    ; instance : string
    ; kind : string
    ; params : (string * int) list
    ; memories : (string * Memory.t) list
    ; inputs : string list
    ; outputs : string list
    }
  [@@deriving sexp_of]

  let of_json json =
    let params = Json.assoc (Json.field json "params") ~f:Json.int in
    List.iter params ~f:(fun (name, value) -> Sv_int.check_int name value);
    { name = Json.string (Json.field json "name")
    ; instance = Json.string (Json.field json "instance")
    ; kind = Json.string (Json.field json "kind")
    ; params
    ; memories = Json.assoc (Json.field json "memories") ~f:Memory.of_json
    ; inputs = Json.list (Json.field json "inputs") ~f:Json.string
    ; outputs = Json.list (Json.field json "outputs") ~f:Json.string
    }
  ;;

  let param ?default t name =
    match List.Assoc.find t.params name ~equal:String.equal, default with
    | Some value, _ | None, Some value -> value
    | None, None -> raise_s [%message "design.json: node has no parameter" t.name name]
  ;;

  let words t name =
    match List.Assoc.find t.memories name ~equal:String.equal with
    | Some memory -> Memory.words memory
    | None -> raise_s [%message "design.json: node has no memory" t.name name]
  ;;
end

module Stream_width = struct
  type t =
    { name : string
    ; width : int
    }
  [@@deriving sexp_of]

  let of_json json =
    { name = Json.string (Json.field json "name")
    ; width = Json.int (Json.field json "width")
    }
  ;;
end

module Mode = struct
  type t =
    | Streaming
    | Fixed of { latency : int }
  [@@deriving sexp_of]

  (* Older files have no mode field and are streaming designs. *)
  let of_json json =
    match Option.map (Json.find json "mode") ~f:Json.string with
    | None | Some "stream" -> Streaming
    | Some "fixed" -> Fixed { latency = Json.int (Json.field json "latency") }
    | Some mode -> raise_s [%message "design.json: unknown mode" mode]
  ;;
end

type t =
  { top : string
  ; mode : Mode.t
  ; input : string
  ; output : string
  ; streams : Stream_width.t list
  ; nodes : Node.t list
  }
[@@deriving sexp_of]

let format = 1

let of_json json =
  let version = Json.int (Json.field json "format") in
  if version <> format
  then raise_s [%message "design.json: unsupported format" (version : int) (format : int)];
  { top = Json.string (Json.field json "top")
  ; mode = Mode.of_json json
  ; input = Json.string (Json.field json "input")
  ; output = Json.string (Json.field json "output")
  ; streams = Json.list (Json.field json "streams") ~f:Stream_width.of_json
  ; nodes = Json.list (Json.field json "nodes") ~f:Node.of_json
  }
;;

let of_string text = of_json (Jsonaf.of_string text)

let width t name =
  match List.find t.streams ~f:(fun stream -> String.equal stream.name name) with
  | Some stream -> stream.width
  | None -> raise_s [%message "design.json: unknown stream" name]
;;
