"""Transfer and program a verified Kria firmware package over SSH."""

import base64
import json
from pathlib import Path
import shlex
import subprocess

from . import kria_install
from .board import BoardError
from .register_transport import ssh_command


RECEIVE = '''
import base64
import sys

packet = json.load(sys.stdin)
with tempfile.TemporaryDirectory() as temporary:
    source = Path(temporary)
    for filename, encoded in packet["files"].items():
        if Path(filename).name != filename or filename in (".", ".."):
            raise ValueError("invalid package filename")
        (source / filename).write_bytes(base64.b64decode(encoded, validate=True))
    install(source, packet["board"], program=True, reuse=True)
'''


def program(host, firmware, board, manifest, timeout=300.0):
    command = ssh_command(host, timeout)
    _, files = kria_install.read_package(firmware, board)
    if files["manifest.json"] != Path(manifest).read_bytes():
        raise BoardError("firmware manifest differs from the compiled model")
    packet = {"board": board,
              "files": {name: base64.b64encode(data).decode("ascii") for name, data in files.items()}}
    script = Path(kria_install.__file__).read_text() + RECEIVE
    remote = "sudo -n python3 -c " + shlex.quote(script)
    completed = subprocess.run(command + [remote], input=json.dumps(packet),
                               text=True, capture_output=True, timeout=timeout)
    if completed.returncode:
        raise BoardError("Kria programming failed:\n%s" % completed.stderr[-2000:].strip())
