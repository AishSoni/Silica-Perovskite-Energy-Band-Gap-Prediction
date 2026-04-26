"""
Download candidate double-perovskite materials from the Materials Project.

The downloader intentionally performs a broad candidate query and writes a
manifest. Structural and chemistry validation happens in src/validate_dataset.py.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from mp_api.client import MPRester

from src.pipeline_config import (
    config_checksum,
    ensure_parent,
    get_api_key,
    load_config,
    path_from_config,
    range_tuple,
)


def _as_dict(doc: Any) -> Dict[str, Any]:
    """Convert a Materials Project summary doc or raw dict to a dictionary."""
    if isinstance(doc, dict):
        return doc
    if hasattr(doc, "model_dump"):
        return doc.model_dump()
    if hasattr(doc, "dict"):
        return doc.dict()
    return dict(vars(doc))


def _serializable(value: Any) -> Any:
    """Convert nested MP/pymatgen objects into JSON-serializable values."""
    if value is None:
        return None
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _serializable(v) for k, v in value.items()}
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return [_serializable(item) for item in value]
    return str(value)


def _json_string(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(_serializable(value), sort_keys=True, default=str)


def flatten_material_record(doc: Any) -> Dict[str, Any]:
    """Flatten the fields needed by the downstream pipeline."""
    raw = {key: _serializable(value) for key, value in _as_dict(doc).items()}

    symmetry = raw.get("symmetry") or {}
    if not isinstance(symmetry, dict):
        symmetry = {}

    elements = raw.get("elements") or []
    element_symbols = []
    for element in elements:
        if isinstance(element, dict):
            symbol = element.get("element") or element.get("symbol") or element.get("name")
        else:
            symbol = str(element)
        if symbol:
            element_symbols.append(symbol)

    structure = raw.get("structure")

    return {
        "material_id": raw.get("material_id"),
        "formula_pretty": raw.get("formula_pretty"),
        "formula_anonymous": raw.get("formula_anonymous"),
        "elements": ",".join(element_symbols),
        "nelements": raw.get("nelements"),
        "band_gap": raw.get("band_gap"),
        "is_gap_direct": raw.get("is_gap_direct"),
        "formation_energy_per_atom": raw.get("formation_energy_per_atom"),
        "energy_above_hull": raw.get("energy_above_hull"),
        "density": raw.get("density"),
        "volume": raw.get("volume"),
        "nsites": raw.get("nsites"),
        "theoretical": raw.get("theoretical"),
        "deprecated": raw.get("deprecated"),
        "energy_per_atom": raw.get("energy_per_atom"),
        "is_stable": raw.get("is_stable"),
        "crystal_system": symmetry.get("crystal_system"),
        "spacegroup_number": symmetry.get("number"),
        "spacegroup_symbol": symmetry.get("symbol"),
        "symmetry_json": _json_string(raw.get("symmetry")),
        "structure_json": _json_string(structure),
    }


def build_search_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build Materials Project summary.search keyword arguments from config."""
    query = config.get("query", {})
    kwargs: Dict[str, Any] = {
        "num_elements": tuple(int(v) for v in range_tuple(query["num_elements"])),
        "num_sites": tuple(int(v) for v in range_tuple(query["num_sites"])),
        "energy_above_hull": range_tuple(query["energy_above_hull"]),
        "band_gap": range_tuple(query["band_gap"]),
        "fields": config["fields"],
    }

    theoretical = query.get("theoretical")
    if theoretical is not None:
        kwargs["theoretical"] = theoretical

    return kwargs


def download_candidates(config: Dict[str, Any], api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """Download candidate records from Materials Project."""
    api_key = api_key or get_api_key()
    search_kwargs = build_search_kwargs(config)
    max_results = config.get("query", {}).get("max_results")

    print("Connecting to Materials Project API...")
    print("Candidate query:")
    for key, value in search_kwargs.items():
        if key != "fields":
            print(f"  {key}: {value}")
    print(f"  fields: {len(search_kwargs['fields'])} requested")

    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(**search_kwargs)

    if max_results:
        docs = docs[: int(max_results)]

    records = [flatten_material_record(doc) for doc in docs]

    if not config.get("query", {}).get("include_deprecated", False):
        records = [record for record in records if not bool(record.get("deprecated"))]

    print(f"Downloaded {len(records)} candidate materials after deprecated filtering.")
    return records


def save_outputs(records: List[Dict[str, Any]], config: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    """Save raw candidate data and manifest."""
    raw_json_path = ensure_parent(path_from_config(config, "raw_json"))
    raw_csv_path = ensure_parent(path_from_config(config, "raw_csv"))
    manifest_path = ensure_parent(path_from_config(config, "manifest"))

    with raw_json_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, default=str)

    pd.DataFrame(records).to_csv(raw_csv_path, index=False)

    material_ids = sorted(str(record.get("material_id")) for record in records if record.get("material_id"))
    material_id_payload = json.dumps(material_ids, sort_keys=True).encode("utf-8")

    import hashlib
    import mp_api

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_checksum": config_checksum(config),
        "dataset_name": config.get("dataset", {}).get("name"),
        "dataset_family": config.get("dataset", {}).get("family"),
        "candidate_count": len(records),
        "material_id_count": len(material_ids),
        "material_id_checksum": hashlib.sha256(material_id_payload).hexdigest(),
        "raw_json": str(raw_json_path),
        "raw_csv": str(raw_csv_path),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "mp_api_version": getattr(mp_api, "__version__", "unknown"),
        "search_kwargs": {k: v for k, v in build_search_kwargs(config).items() if k != "fields"},
        "fields": config.get("fields", []),
    }

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=str)

    print(f"Saved raw JSON: {raw_json_path}")
    print(f"Saved raw CSV: {raw_csv_path}")
    print(f"Saved manifest: {manifest_path}")
    return manifest


def run_download(config_path: str | Path = "experiments/query_config.yaml") -> Dict[str, Any]:
    """Run the configured candidate download."""
    config_path = Path(config_path)
    config = load_config(config_path)
    records = download_candidates(config)
    return save_outputs(records, config, config_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download candidate double-perovskite Materials Project records.")
    parser.add_argument("--config", default="experiments/query_config.yaml", help="Path to canonical query config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_download(args.config)


if __name__ == "__main__":
    main()

