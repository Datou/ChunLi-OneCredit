"""Validate and gzip a raw Gym Retro Genesis Plus GX 1.7.5 state."""
import argparse
import gzip
from pathlib import Path

HEADER = b"GENPLUS-GX 1.7.5"
STATE_SIZE = 1_036_288


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--already-gzip", action="store_true", help="Require gzip input")
    args = parser.parse_args()
    try:
        data = args.input.read_bytes()
        compressed = data.startswith(b"\x1f\x8b")
        if args.already_gzip and not compressed:
            parser.error("expected gzip input")
        payload = gzip.decompress(data) if compressed else data
    except (OSError, EOFError) as error:
        parser.error(str(error))

    if len(payload) != STATE_SIZE or payload[:16] != HEADER:
        parser.error(
            f"expected Genesis Plus GX 1.7.5, {STATE_SIZE} bytes after decompression "
            f"(got {payload[:16]!r}, {len(payload)} bytes)"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data if compressed else gzip.compress(payload, compresslevel=6, mtime=0))
    print(f"{args.output}: payload verified and saved")


if __name__ == "__main__":
    main()
