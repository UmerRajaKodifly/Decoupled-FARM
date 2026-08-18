#!/usr/bin/env python3
"""Build query_index.json for viewer / API retrieval."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from build_query_index import build_query_index, load_vocab, write_query_index  # noqa: E402
from caption import load_scene  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("build_index")


def main() -> int:
    repo = _HERE.parent
    p = argparse.ArgumentParser(description="Build query index JSON")
    p.add_argument("--scene-state", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--vocab-file", type=Path, default=repo / "vocab" / "construction_vocab.txt")
    args = p.parse_args()

    if not args.scene_state.is_file():
        log.error("Missing %s", args.scene_state)
        return 2

    out = args.output or (args.scene_state.parent / "query_index.json")
    ss = load_scene(args.scene_state)
    vocab = load_vocab(args.vocab_file)
    index = build_query_index(ss, vocab)
    write_query_index(out, index)
    log.info("Wrote %s (%d objects)", out, index["n_objects"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
