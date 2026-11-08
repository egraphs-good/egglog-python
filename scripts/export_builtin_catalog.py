"""Export the compiled transport's native catalog without importing egglog."""

import argparse
import hashlib
import importlib.machinery
import importlib.util
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True, help="compiled bindings extension")
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parents[1] / "python/egglog/_builtin_catalog.pb"
    )
    parser.add_argument("--check", action="store_true", help="compare against the packaged catalog")
    args = parser.parse_args()
    extension_path = args.extension.resolve()
    loader = importlib.machinery.ExtensionFileLoader("bindings", str(extension_path))
    spec = importlib.util.spec_from_file_location("bindings", extension_path, loader=loader)
    if spec is None:
        parser.error("Cannot load the compiled bindings extension")
    extension = importlib.util.module_from_spec(spec)
    loader.exec_module(extension)
    # The default GetBuiltinsRequest has an empty protobuf encoding.
    encoded = extension._ProtoEngine().get_builtins(b"")
    if args.check:
        if args.output.read_bytes() != encoded:
            parser.error("Packaged catalog differs from the native export; export it again")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(encoded)
    print(f"SHA-256 {hashlib.sha256(encoded).hexdigest()}")


if __name__ == "__main__":
    main()
