#!/usr/bin/env python3
"""
Convert endpoint summary JSON files into NDJSON rows that can be used by the
power analysis tooling (which expects JSONL/NDJSON input).

Usage:
    # Convert a single file
    python scripts/export_endpoints_to_ndjson.py \
        --input outputs/endpoints/endpoints_caffeine_gpt2_20251118_212634.json \
        --output outputs/endpoints/pilot_data.jsonl

    # Convert every endpoints_*.json in a directory
    python scripts/export_endpoints_to_ndjson.py \
        --input-dir outputs/endpoints \
        --output outputs/endpoints/pilot_data.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List


def load_summary(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def add_record(
    records: List[Dict[str, Any]],
    *,
    timestamp: str,
    model_name: str,
    endpoint_name: str,
    endpoint_type: str,
    item_id: str,
    pack: str,
    condition: str,
    score: float,
):
    """Append a single NDJSON row."""
    if score is None:
        return
    records.append(
        {
            "timestamp": timestamp,
            "model": model_name,
            "endpoint": endpoint_name,
            "endpoint_type": endpoint_type,
            "item_id": item_id,
            "pack": pack,
            "condition": condition,
            "score": score,
        }
    )


def extract_records(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert endpoint summary into flat rows."""
    records: List[Dict[str, Any]] = []
    timestamp = summary.get("timestamp")
    model_name = summary.get("model_name")
    pack_name = summary.get("pack_name")

    def process_endpoint(endpoints: Dict[str, Any], endpoint_type: str):
        for endpoint_name, data in endpoints.items():
            details = data.get("details", {})
            components = details.get("components", [])
            treatment_components = details.get("treatment_components", {})
            baseline_components = details.get("baseline_components", {})
            placebo_components = details.get("placebo_components", {})

            # Component-level rows (if available)
            for comp in components:
                metric_key = f"{comp['test']}.{comp['subscale']}"

                add_record(
                    records,
                    timestamp=timestamp,
                    model_name=model_name,
                    endpoint_name=endpoint_name,
                    endpoint_type=endpoint_type,
                    item_id=f"{endpoint_name}.{metric_key}",
                    pack=pack_name,
                    condition="treatment",
                    score=treatment_components.get(metric_key),
                )

                add_record(
                    records,
                    timestamp=timestamp,
                    model_name=model_name,
                    endpoint_name=endpoint_name,
                    endpoint_type=endpoint_type,
                    item_id=f"{endpoint_name}.{metric_key}",
                    pack=pack_name,
                    condition="baseline",
                    score=baseline_components.get(metric_key),
                )

                add_record(
                    records,
                    timestamp=timestamp,
                    model_name=model_name,
                    endpoint_name=endpoint_name,
                    endpoint_type=endpoint_type,
                    item_id=f"{endpoint_name}.{metric_key}",
                    pack=pack_name,
                    condition="placebo",
                    score=placebo_components.get(metric_key),
                )

            # Aggregate rows (always include)
            add_record(
                records,
                timestamp=timestamp,
                model_name=model_name,
                endpoint_name=endpoint_name,
                endpoint_type=endpoint_type,
                item_id=f"{endpoint_name}.aggregate",
                pack=pack_name,
                condition="treatment",
                score=data.get("treatment_score"),
            )
            add_record(
                records,
                timestamp=timestamp,
                model_name=model_name,
                endpoint_name=endpoint_name,
                endpoint_type=endpoint_type,
                item_id=f"{endpoint_name}.aggregate",
                pack=pack_name,
                condition="baseline",
                score=data.get("baseline_score"),
            )
            add_record(
                records,
                timestamp=timestamp,
                model_name=model_name,
                endpoint_name=endpoint_name,
                endpoint_type=endpoint_type,
                item_id=f"{endpoint_name}.aggregate",
                pack=pack_name,
                condition="placebo",
                score=data.get("placebo_score"),
            )

    process_endpoint(summary.get("primary_endpoints", {}), "primary")
    process_endpoint(summary.get("secondary_endpoints", {}), "secondary")

    # Remove rows without score
    return [row for row in records if row.get("score") is not None]


def write_ndjson(records: List[Dict[str, Any]], output_path: Path, append: bool = False):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with output_path.open(mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record))
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert endpoint summary JSON into NDJSON rows."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="Single endpoint JSON file to convert")
    group.add_argument(
        "--input-dir",
        help="Directory containing endpoint summary JSON files (e.g., outputs/endpoints)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination NDJSON/JSONL file (will be overwritten unless --append is set)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to output file instead of overwriting",
    )

    args = parser.parse_args()
    output_path = Path(args.output)

    # Determine input files
    input_files: List[Path] = []
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        input_files = [input_path]
    else:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        # Grab endpoint summary files
        input_files = sorted(
            f
            for f in input_dir.glob("endpoints_*.json")
            if f.is_file()
        )
        if not input_files:
            raise RuntimeError(
                f"No endpoint summary JSON files found in {input_dir} "
                "(looked for endpoints_*.json)."
            )

    total_written = 0
    append = args.append

    for idx, input_path in enumerate(input_files):
        summary = load_summary(input_path)
        records = extract_records(summary)

        if not records:
            print(
                f"[WARN] No component data found in {input_path}. "
                "Skipping file."
            )
            continue

        write_ndjson(records, output_path, append=append or idx > 0)
        total_written += len(records)
        print(
            f"[OK] Processed {input_path.name}: wrote {len(records)} rows "
            f"to {output_path}"
        )

    if total_written == 0:
        raise RuntimeError(
            "No records were written. Ensure endpoint JSON files contain data."
        )

    print(
        f"[DONE] Total rows written: {total_written} -> {output_path}"
        f"{' (append mode)' if args.append else ''}"
    )


if __name__ == "__main__":
    main()


