"""
Package an actual native catalog export without importing the Python frontend.

First run egglog-experimental's export_builtin_catalog example from the pinned
native checkout. This command consumes that binary snapshot; it never discovers
signatures through an EGraph during import or authoring. --check is a byte-for-
byte reproducibility gate, so changed native records cannot be silently hidden.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from egglog_proto.egglog.v1 import egglog_pb as pb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parents[1] / "python/egglog/_builtin_catalog.pb"
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    encoded = args.export.read_bytes()
    program = pb.Program.from_binary(encoded)
    if program.ir_version != 1 or program.commands or program.rules or program.rulesets:
        parser.error("Expected a version-1 declaration-only native catalog")
    names: set[tuple[str, str]] = set()
    for declaration in program.declarations:
        kind = declaration.kind
        if kind is None or kind.field not in {"host_sort_family", "host_primitive"}:
            parser.error("Native catalog may contain only host family/primitive declarations")
        namespace = "sort" if kind.field == "host_sort_family" else "callable"
        key = namespace, kind.value.name
        if not key[1] or key in names:
            parser.error(f"Duplicate or empty native definition: {key}")
        names.add(key)
    if args.check:
        if args.output.read_bytes() != encoded:
            parser.error("Packaged catalog differs from the native export; regenerate it")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(encoded)
    print(f"{len(names)} native definitions; SHA-256 {hashlib.sha256(encoded).hexdigest()}")


if __name__ == "__main__":
    main()
