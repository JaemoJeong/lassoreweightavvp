from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def safe_name(path: Path) -> str:
    return "__".join(path.with_suffix("").parts) + path.suffix


def make_link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        rel_src = os.path.relpath(src, start=dst.parent)
        dst.symlink_to(rel_src)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"unknown mode: {mode}")


def rebuild_index(results_root: Path, out_dir: Path, mode: str) -> list[tuple[Path, Path]]:
    results_root = results_root.resolve()
    out_dir = out_dir.resolve()
    if results_root == out_dir or results_root not in out_dir.parents:
        raise ValueError(f"out_dir must be inside results_root: {out_dir} not under {results_root}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs: list[tuple[Path, Path]] = []
    for src in sorted(results_root.rglob("*.png")):
        if out_dir in src.resolve().parents:
            continue
        rel = src.relative_to(results_root)
        flat_dst = out_dir / "flat" / safe_name(rel)
        tree_dst = out_dir / "tree" / rel
        make_link_or_copy(src, flat_dst, mode)
        make_link_or_copy(src, tree_dst, mode)
        pairs.append((src, flat_dst))
    return pairs


def write_manifest(out_dir: Path, results_root: Path, pairs: list[tuple[Path, Path]], mode: str) -> None:
    manifest = out_dir / "manifest.tsv"
    with manifest.open("w") as f:
        f.write("index_path\tsource_path\n")
        for src, dst in pairs:
            f.write(f"{dst.relative_to(out_dir)}\t{src.relative_to(results_root)}\n")

    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# PNG Result Index",
                "",
                f"Mode: `{mode}`",
                f"PNG count: `{len(pairs)}`",
                "",
                "- `flat/`: all PNGs in one directory with path-safe names.",
                "- `tree/`: PNGs grouped by their original result path.",
                "- `manifest.tsv`: mapping from indexed PNG to original result path.",
                "",
                "This directory is disposable. Removing it does not remove original result files when mode is `symlink`.",
                "",
            ]
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a disposable PNG index for result folders.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    args = parser.parse_args()

    results_root = args.results_root
    out_dir = args.out_dir or (results_root / "_png_index")
    pairs = rebuild_index(results_root=results_root, out_dir=out_dir, mode=args.mode)
    write_manifest(out_dir=out_dir.resolve(), results_root=results_root.resolve(), pairs=pairs, mode=args.mode)
    print(f"Indexed {len(pairs)} PNGs into {out_dir}")
    print(f"Flat view: {out_dir / 'flat'}")
    print(f"Tree view: {out_dir / 'tree'}")


if __name__ == "__main__":
    main()
