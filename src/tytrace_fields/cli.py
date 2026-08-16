from __future__ import annotations

import argparse
from pathlib import Path

from .exporter import export_typhoon


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export TyTrace weather fields to GeoJSON")
    parser.add_argument("--typhoon", action="append", required=True, help="dataset directory name")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../TyTrace/public/weatherdata"),
        help="frontend weatherdata directory",
    )
    parser.add_argument("--resolution", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    for typhoon in arguments.typhoon:
        result = export_typhoon(
            arguments.dataset_root,
            arguments.output,
            typhoon,
            resolution=arguments.resolution,
            overwrite=arguments.overwrite,
        )
        print(f"exported {result.typhoon}: {result.frame_count} frames -> {result.output_directory}")


if __name__ == "__main__":
    main()

