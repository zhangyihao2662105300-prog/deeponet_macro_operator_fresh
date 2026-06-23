#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Print a Markdown table from a v2 multi-case compact summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FIELDS = (
    "case_id",
    "strict_v2_coordinate_pass",
    "q_useful_removed_rigid_rel",
    "B_rigid_residual_rel",
    "B_local_chain_rule_projected_rel",
    "B_local_chain_rule_raw_rel",
    "LE_local_roundtrip_rel",
    "local_frame_orthonormal_max",
)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, help="Path to v2b_multi_case_summary.json")
    args = parser.parse_args()

    path = Path(args.summary).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("case_summaries", [])
    print(f"summary: `{path}`")
    print(f"compact_count: {payload.get('compact_count')}")
    print(f"pass_count: {payload.get('pass_count')}")
    print(f"fail_count: {payload.get('fail_count')}")
    print()
    print("| " + " | ".join(FIELDS) + " |")
    print("|" + "|".join("---" for _ in FIELDS) + "|")
    for row in rows:
        print("| " + " | ".join(fmt(row.get(field)) for field in FIELDS) + " |")


if __name__ == "__main__":
    main()
