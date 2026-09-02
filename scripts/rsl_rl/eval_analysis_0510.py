#!/usr/bin/env python3
"""Analyze evaluation results and generate summary reports (0510 Delta IK version).

Usage:
    python eval_analysis_0510.py --input eval_results/20260428_221000/all_trials.csv \
                                 --output_dir eval_results/20260428_221000
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

# ensure local dir is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_config_0510 import classify_mass, classify_position, classify_height, classify_failure


def load_csv(path: str) -> list[dict]:
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            # Type conversions
            row["trial_id"] = int(row["trial_id"])
            row["object_mass"] = float(row["object_mass"])
            row["obj_init_x"] = float(row["obj_init_x"])
            row["obj_init_y"] = float(row["obj_init_y"])
            row["obj_init_z"] = float(row["obj_init_z"])
            row["target_x"] = float(row["target_x"])
            row["target_y"] = float(row["target_y"])
            row["target_z"] = float(row["target_z"])
            row["episode_length"] = int(row["episode_length"])
            row["terminated_early"] = row["terminated_early"] == "True"
            row["reach_success"] = row["reach_success"] == "True"
            row["lift_success"] = row["lift_success"] == "True"
            row["goal_track_success"] = row["goal_track_success"] == "True"
            row["max_consecutive_reach"] = int(row["max_consecutive_reach"])
            row["max_consecutive_lift"] = int(row["max_consecutive_lift"])
            row["max_consecutive_goal_track"] = int(row["max_consecutive_goal_track"])
            row["min_ee_bbox_dist"] = float(row["min_ee_bbox_dist"])
            row["max_object_height"] = float(row["max_object_height"])
            row["min_goal_dist"] = float(row["min_goal_dist"])
            rows.append(row)
    return rows


def compute_rates(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {"trials": 0, "reach_pct": 0, "lift_pct": 0, "goal_pct": 0}
    return {
        "trials": n,
        "reach_pct": round(sum(1 for r in records if r["reach_success"]) / n * 100, 1),
        "lift_pct": round(sum(1 for r in records if r["lift_success"]) / n * 100, 1),
        "goal_pct": round(sum(1 for r in records if r["goal_track_success"]) / n * 100, 1),
    }


def format_table(title: str, headers: list[str], rows: list[list], col_widths: list[int] = None) -> str:
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            max_w = len(str(h))
            for row in rows:
                max_w = max(max_w, len(str(row[i])))
            col_widths.append(max_w + 2)

    lines = [f"\n=== {title} ===\n"]

    # Header
    header_line = ""
    for h, w in zip(headers, col_widths):
        header_line += str(h).ljust(w)
    lines.append(header_line)
    lines.append("-" * len(header_line))

    # Data rows
    for row in rows:
        line = ""
        for val, w in zip(row, col_widths):
            line += str(val).ljust(w)
        lines.append(line)

    return "\n".join(lines)


def stratify(records: list[dict], key_fn) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for r in records:
        groups[key_fn(r)].append(r)
    return dict(groups)


def generate_report(records: list[dict]) -> str:
    sections = []

    # 1. Overall
    overall = compute_rates(records)
    by_group = stratify(records, lambda r: r["object_group"])
    table_rows = []
    for g in ["train", "test"]:
        if g in by_group:
            rates = compute_rates(by_group[g])
            table_rows.append([g.title(), rates["trials"], f"{rates['reach_pct']}%",
                               f"{rates['lift_pct']}%", f"{rates['goal_pct']}%"])
    table_rows.append(["Total", overall["trials"], f"{overall['reach_pct']}%",
                        f"{overall['lift_pct']}%", f"{overall['goal_pct']}%"])
    sections.append(format_table("整體結果", ["Group", "Trials", "Reach%", "Lift%", "GoalTrack%"], table_rows))

    # 2. By Object ID
    by_obj = stratify(records, lambda r: (r["object_group"], r["object_id"]))
    table_rows = []
    for (group, obj_id), recs in sorted(by_obj.items()):
        rates = compute_rates(recs)
        table_rows.append([group, obj_id, rates["trials"], f"{rates['reach_pct']}%",
                           f"{rates['lift_pct']}%", f"{rates['goal_pct']}%"])
    sections.append(format_table("按物體 ID", ["Group", "ObjectID", "Trials", "Reach%", "Lift%", "GoalTrack%"], table_rows))

    # 3. By Shape Class
    by_shape = stratify(records, lambda r: r["object_shape_class"])
    table_rows = []
    for shape in ["flat_long", "cup_bowl", "cylinder", "irregular"]:
        if shape in by_shape:
            rates = compute_rates(by_shape[shape])
            table_rows.append([shape, rates["trials"], f"{rates['reach_pct']}%",
                               f"{rates['lift_pct']}%", f"{rates['goal_pct']}%"])
    sections.append(format_table("按物體形狀分類", ["Shape", "Trials", "Reach%", "Lift%", "GoalTrack%"], table_rows))

    # 4. By Mass
    by_mass = stratify(records, lambda r: classify_mass(r["object_mass"]))
    table_rows = []
    for m in ["light", "medium", "heavy"]:
        if m in by_mass:
            rates = compute_rates(by_mass[m])
            table_rows.append([m, rates["trials"], f"{rates['reach_pct']}%",
                               f"{rates['lift_pct']}%", f"{rates['goal_pct']}%"])
    sections.append(format_table("按質量區間", ["Mass", "Trials", "Reach%", "Lift%", "GoalTrack%"], table_rows))

    # 5. By Position
    by_pos = stratify(records, lambda r: r["obj_position_class"])
    table_rows = []
    for p in ["left", "center", "right"]:
        if p in by_pos:
            rates = compute_rates(by_pos[p])
            table_rows.append([p, rates["trials"], f"{rates['reach_pct']}%",
                               f"{rates['lift_pct']}%", f"{rates['goal_pct']}%"])
    sections.append(format_table("按物體初始方位", ["Position", "Trials", "Reach%", "Lift%", "GoalTrack%"], table_rows))

    # 6. By Target Height
    by_height = stratify(records, lambda r: r["target_height_class"])
    table_rows = []
    for h in ["very_low", "low", "mid", "high"]:
        if h in by_height:
            rates = compute_rates(by_height[h])
            table_rows.append([h, rates["trials"], f"{rates['reach_pct']}%",
                               f"{rates['lift_pct']}%", f"{rates['goal_pct']}%"])
    sections.append(format_table("按目標高度區間", ["Height", "Trials", "Reach%", "Lift%", "GoalTrack%"], table_rows))

    # 7. Failure mode distribution (Train vs Test)
    by_group = stratify(records, lambda r: r["object_group"])
    train_recs = by_group.get("train", [])
    test_recs = by_group.get("test", [])
    total_recs = records

    n_train = len(train_recs)
    n_test = len(test_recs)
    n_total = len(total_recs)

    train_counts = defaultdict(int)
    for r in train_recs:
        train_counts[r["failure_mode"]] += 1

    test_counts = defaultdict(int)
    for r in test_recs:
        test_counts[r["failure_mode"]] += 1

    total_counts = defaultdict(int)
    for r in total_recs:
        total_counts[r["failure_mode"]] += 1

    modes = ["success", "transport_fail", "drop_after_lift", "grasp_fail",
             "approach_only", "no_approach", "out_of_bounds"]
    table_rows = []
    for mode in modes:
        if total_counts[mode] > 0:
            c_tr = train_counts[mode]
            pct_tr = f"{c_tr / n_train * 100:.1f}%" if n_train > 0 else "0.0%"
            tr_str = f"{c_tr} ({pct_tr})"

            c_te = test_counts[mode]
            pct_te = f"{c_te / n_test * 100:.1f}%" if n_test > 0 else "0.0%"
            te_str = f"{c_te} ({pct_te})"

            c_tot = total_counts[mode]
            pct_tot = f"{c_tot / n_total * 100:.1f}%" if n_total > 0 else "0.0%"
            tot_str = f"{c_tot} ({pct_tot})"

            table_rows.append([mode, tr_str, te_str, tot_str])
    headers = ["FailureMode", f"Train (N={n_train})", f"Test (N={n_test})", f"Total (N={n_total})"]
    sections.append(format_table("失敗原因分布", headers, table_rows))

    return "\n\n".join(sections) + "\n"


def generate_json_summary(records: list[dict]) -> dict:
    """Generate machine-readable summary."""
    summary = {
        "total": compute_rates(records),
        "by_group": {},
        "by_object": {},
        "by_shape": {},
        "by_position": {},
        "by_height": {},
        "by_mass": {},
        "by_failure_mode": {},
    }

    for group, recs in stratify(records, lambda r: r["object_group"]).items():
        summary["by_group"][group] = compute_rates(recs)

    for (group, obj_id), recs in stratify(records, lambda r: (r["object_group"], r["object_id"])).items():
        summary["by_object"][f"{group}/{obj_id}"] = compute_rates(recs)

    for shape, recs in stratify(records, lambda r: r["object_shape_class"]).items():
        summary["by_shape"][shape] = compute_rates(recs)

    for pos, recs in stratify(records, lambda r: r["obj_position_class"]).items():
        summary["by_position"][pos] = compute_rates(recs)

    for h, recs in stratify(records, lambda r: r["target_height_class"]).items():
        summary["by_height"][h] = compute_rates(recs)

    for m, recs in stratify(records, lambda r: classify_mass(r["object_mass"])).items():
        summary["by_mass"][m] = compute_rates(recs)

    modes = ["success", "transport_fail", "drop_after_lift", "grasp_fail",
             "approach_only", "no_approach", "out_of_bounds"]
    by_group = stratify(records, lambda r: r["object_group"])
    for g, recs in by_group.items():
        cnts = defaultdict(int)
        for r in recs: cnts[r["failure_mode"]] += 1
        summary["by_failure_mode"][g] = {m: cnts[m] for m in modes if cnts[m] > 0}
    
    cnts_tot = defaultdict(int)
    for r in records: cnts_tot[r["failure_mode"]] += 1
    summary["by_failure_mode"]["total"] = {m: cnts_tot[m] for m in modes if cnts_tot[m] > 0}

    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze 0510 evaluation results.")
    parser.add_argument("--input", type=str, required=True, help="Path to all_trials.csv")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: same as input)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.input)

    print(f"[ANALYSIS 0510] Loading {args.input}")
    records = load_csv(args.input)
    print(f"[ANALYSIS 0510] Loaded {len(records)} trial records")

    # Text report
    report = generate_report(records)
    txt_path = os.path.join(args.output_dir, "summary.txt")
    with open(txt_path, "w") as f:
        f.write(report)
    print(report)
    print(f"[ANALYSIS 0510] Summary → {txt_path}")

    # JSON summary
    summary = generate_json_summary(records)
    json_path = os.path.join(args.output_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[ANALYSIS 0510] JSON → {json_path}")


if __name__ == "__main__":
    main()
