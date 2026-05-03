from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


SKIP_NAMES = {"_organized", "_png_index"}


def classify(path: Path) -> str | None:
    name = path.name
    if name in SKIP_NAMES or name.startswith("."):
        return None

    if path.is_file():
        suffix = path.suffix.lower()
        if suffix in {".png", ".pdf"}:
            return "figures"
        if suffix in {".md", ".txt"}:
            return "docs"
        return "misc"

    if name.startswith("sweep_lambda"):
        return "sweeps/lambda"
    if name.startswith("sweep_kappa_eta") or name.startswith("grid_kappa_eta"):
        return "sweeps/kappa_eta"
    if name.startswith("tune"):
        return "tuning"
    if name.endswith("_details") or "_details" in name or name.startswith("best_"):
        return "details"
    if name in {"external_mean", "mean_source_comparison"}:
        return "mean_sources"
    if name.startswith("lam"):
        return "single_runs"
    return "misc"


def link_name(path: Path) -> str:
    return path.name


def make_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    rel_src = os.path.relpath(src.resolve(), start=dst.parent.resolve())
    dst.symlink_to(rel_src, target_is_directory=src.is_dir())


def rebuild(results_root: Path, out_dir: Path) -> list[tuple[Path, Path, str]]:
    results_root = results_root.resolve()
    out_dir = out_dir.resolve()
    if results_root == out_dir or results_root not in out_dir.parents:
        raise ValueError(f"out_dir must be inside results_root: {out_dir} not under {results_root}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries: list[tuple[Path, Path, str]] = []
    for src in sorted(results_root.iterdir(), key=lambda p: p.name):
        group = classify(src)
        if group is None:
            continue
        dst = out_dir / group / link_name(src)
        make_symlink(src, dst)
        entries.append((src, dst, group))

    png_index = results_root / "_png_index"
    if png_index.exists():
        dst = out_dir / "png_index"
        make_symlink(png_index, dst)
        entries.append((png_index, dst, "png_index"))

    return entries


def write_manifest(out_dir: Path, results_root: Path, entries: list[tuple[Path, Path, str]]) -> None:
    with (out_dir / "manifest.tsv").open("w") as f:
        f.write("group\tindex_path\tsource_path\n")
        for src, dst, group in entries:
            f.write(f"{group}\t{dst.relative_to(out_dir)}\t{src.relative_to(results_root)}\n")

    group_counts: dict[str, int] = {}
    for _, _, group in entries:
        group_counts[group] = group_counts.get(group, 0) + 1

    lines = [
        "# Organized Result Index",
        "",
        "This is a disposable symlink index. Removing `_organized` does not remove original result files.",
        "",
        "Groups:",
    ]
    for group in sorted(group_counts):
        lines.append(f"- `{group}`: {group_counts[group]}")
    lines.extend(
        [
            "",
            "Suggested entry points:",
            "- `sweeps/lambda/`: lambda sweeps",
            "- `sweeps/kappa_eta/`: kappa/eta grids",
            "- `tuning/`: hyperparameter tuning runs",
            "- `details/`: runs with segment detail text files",
            "- `png_index/`: all PNGs grouped separately",
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a grouped symlink index for result artifacts.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    results_root = args.results_root
    out_dir = args.out_dir or (results_root / "_organized")
    entries = rebuild(results_root=results_root, out_dir=out_dir)
    write_manifest(out_dir=out_dir.resolve(), results_root=results_root.resolve(), entries=entries)
    print(f"Indexed {len(entries)} result artifacts into {out_dir}")
    print(f"Manifest: {out_dir / 'manifest.tsv'}")


if __name__ == "__main__":
    main()
