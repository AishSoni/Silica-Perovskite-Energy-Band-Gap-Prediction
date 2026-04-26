"""
Dataset validation gate for candidate double-perovskite records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from .pipeline_config import ensure_dir, ensure_parent, load_config, path_from_config


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _as_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _split_elements(value: Any) -> List[str]:
    if value is None or pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _structure_present(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    text = str(value).strip()
    return text not in {"", "{}", "[]", "None", "nan"}


def validate_record(record: Dict[str, Any], config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a single candidate record and return acceptance plus reasons."""
    reasons: List[str] = []
    validation = config["validation"]
    query = config["query"]
    chemistry = config["chemistry"]

    formula_anonymous = str(record.get("formula_anonymous") or "").strip()
    accepted_patterns = set(validation.get("accepted_formula_anonymous", []))
    if formula_anonymous not in accepted_patterns:
        reasons.append("formula_anonymous_not_accepted")

    elements = set(_split_elements(record.get("elements")))
    x_site = set(chemistry.get("x_site", []))
    a_or_b_site = set(chemistry.get("a_site", [])) | set(chemistry.get("b_site", []))

    if validation.get("require_x_site", True) and not (elements & x_site):
        reasons.append("missing_x_site_element")

    if validation.get("require_a_or_b_site", True) and not (elements & a_or_b_site):
        reasons.append("missing_a_or_b_site_element")

    if validation.get("require_band_gap", True) and _as_float(record.get("band_gap")) is None:
        reasons.append("missing_band_gap")

    if validation.get("require_structure", True) and not _structure_present(record.get("structure_json")):
        reasons.append("missing_structure")

    if validation.get("require_not_deprecated", True) and _as_bool(record.get("deprecated")):
        reasons.append("deprecated_material")

    e_hull = _as_float(record.get("energy_above_hull"))
    max_hull = query.get("energy_above_hull", {}).get("max")
    if max_hull is not None and (e_hull is None or e_hull > float(max_hull)):
        reasons.append("energy_above_hull_out_of_range")

    nsites = _as_float(record.get("nsites"))
    min_sites = query.get("num_sites", {}).get("min")
    max_sites = query.get("num_sites", {}).get("max")
    if nsites is None:
        reasons.append("missing_nsites")
    elif min_sites is not None and nsites < float(min_sites):
        reasons.append("too_few_sites")
    elif max_sites is not None and nsites > float(max_sites):
        reasons.append("too_many_sites")

    material_id = record.get("material_id")
    if not material_id:
        reasons.append("missing_material_id")

    return not reasons, reasons


def validate_dataframe(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Validate a candidate dataframe and return accepted/rejected rows plus summary."""
    accepted_rows = []
    rejected_rows = []
    reason_counter: Counter[str] = Counter()
    seen_material_ids = set()

    for row in df.to_dict(orient="records"):
        accepted, reasons = validate_record(row, config)
        material_id = row.get("material_id")

        if accepted and material_id in seen_material_ids:
            accepted = False
            reasons = ["duplicate_material_id"]

        if accepted:
            seen_material_ids.add(material_id)
            accepted_rows.append(row)
        else:
            for reason in reasons:
                reason_counter[reason] += 1
            rejected = dict(row)
            rejected["rejection_reasons"] = ";".join(reasons)
            rejected_rows.append(rejected)

    accepted_df = pd.DataFrame(accepted_rows)
    rejected_df = pd.DataFrame(rejected_rows)

    material_ids = sorted(str(mid) for mid in accepted_df.get("material_id", pd.Series(dtype=str)).dropna().tolist())
    checksum = hashlib.sha256(json.dumps(material_ids, sort_keys=True).encode("utf-8")).hexdigest()

    summary = {
        "candidate_count": int(len(df)),
        "accepted_count": int(len(accepted_df)),
        "rejected_count": int(len(rejected_df)),
        "rejection_reasons": dict(sorted(reason_counter.items())),
        "accepted_material_id_checksum": checksum,
        "accepted_formula_count": int(accepted_df["formula_pretty"].nunique()) if "formula_pretty" in accepted_df else 0,
    }

    return accepted_df, rejected_df, summary


def run_validation(config_path: str | Path = "experiments/query_config.yaml") -> Dict[str, Any]:
    """Validate configured raw candidates and save validation artifacts."""
    config = load_config(config_path)
    raw_csv = path_from_config(config, "raw_csv")
    validated_csv = ensure_parent(path_from_config(config, "validated_csv"))
    validation_dir = ensure_dir(config["paths"]["validation_dir"])

    if not raw_csv.exists():
        raise FileNotFoundError(f"Raw candidate CSV not found: {raw_csv}. Run download_data.py first.")

    df = pd.read_csv(raw_csv)
    accepted_df, rejected_df, summary = validate_dataframe(df, config)

    accepted_df.to_csv(validated_csv, index=False)
    rejected_path = validation_dir / "rejected_materials.csv"
    rejected_df.to_csv(rejected_path, index=False)

    material_ids_path = validation_dir / "accepted_material_ids.txt"
    material_ids = accepted_df.get("material_id", pd.Series(dtype=str)).dropna().astype(str).sort_values()
    material_ids_path.write_text("\n".join(material_ids.tolist()) + ("\n" if len(material_ids) else ""), encoding="utf-8")

    summary_path = validation_dir / "validation_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)

    print(f"Accepted materials: {summary['accepted_count']}")
    print(f"Rejected materials: {summary['rejected_count']}")
    print(f"Saved validated data: {validated_csv}")
    print(f"Saved validation summary: {summary_path}")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate downloaded double-perovskite candidates.")
    parser.add_argument("--config", default="experiments/query_config.yaml", help="Path to canonical query config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_validation(args.config)


if __name__ == "__main__":
    main()

