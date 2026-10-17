"""Regenerate aggregate scorecards from private paired summary files."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.monitor.cli import new_output, write_new_json
from src.monitor.disagreement import strict_json
from .harness import aggregate


def scorecard(comparisons, metadata):
    controls = [row for row in comparisons if row["scenario"] == "stationary_benign"]
    return {"schema_version": 1, "metadata": metadata, "episodes": comparisons,
        "aggregate": aggregate(comparisons),
        "negative_control": {"episodes": len(controls), "alarm_episodes": sum(row["negative_control_alarm"] for row in controls),
            "observed_silent": bool(controls) and not any(row["negative_control_alarm"] for row in controls),
            "proves_permanent_silence": False},
        "limits": [
            "Source-labelled harmful-request policy, not universal toxicity or production-agent safety.",
            "Coverage is prediction-set membership, not accuracy or probability calibration.",
            "Review/abstention counts and unsafe-label allows must accompany coverage.",
            "Silent score scaling is a deliberately simulated positive-control output fault.",
            "Approval and label timing are simulated offline; no production approval service exists.",
            "Episodes reuse published prompt pools and are not independent production trials.",
            "Finite negative controls do not prove permanent silence or universal false-alarm control.",
            "Classifier weights remain unchanged; no automatic repair guarantee exists."]}


def _percent(value):
    return "N/A" if value is None else f"{100 * value:.2f}%"


def markdown(card):
    lines = ["# Paired replay scorecard", "", "Source-labelled offline experiment. No production safety guarantee.", "",
        "| Scenario | Seed | Detection delay (steps) | Response | A/B unsafe-class coverage | A/B benign-class coverage | A/B review rate | A/B unsafe allows | Coverage status |",
        "|---|---:|---:|---|---|---|---|---|---|"]
    for row in card["episodes"]:
        a, b = row["A"]["final_holdout"], row["B"]["final_holdout"]
        latency = row["detection_latency_steps"] if row["detection_latency_steps"] is not None else "N/A"
        lines.append(f"| {row['scenario']} | {row['seed']} | {latency} | {row['response_status']} | "
            f"{_percent(a['by_class']['1']['coverage'])} / {_percent(b['by_class']['1']['coverage'])} | "
            f"{_percent(a['by_class']['0']['coverage'])} / {_percent(b['by_class']['0']['coverage'])} | "
            f"{_percent(a['review_rate'])} / {_percent(b['review_rate'])} | "
            f"{_percent(a['unsafe_label_allow_rate'])} / {_percent(b['unsafe_label_allow_rate'])} | {row['restoration_status']} |")
    control = card["negative_control"]
    lines += ["", f"Negative control: {control['alarm_episodes']} alarm episodes out of {control['episodes']}.",
        "This is an observed finite result, not proof of permanent silence.", "", "## Aggregate", ""]
    for scenario, values in card["aggregate"].items():
        lines += [f"### {scenario}", "", f"Episodes: {values['episodes']}.",
            f"Detected intended changes: {values['detected_intended_changes']}.",
            f"Pre-shift alarm episodes: {values['pre_shift_alarm_episodes']}.",
            f"Median detected latency: {values['median_detected_latency_steps']} steps.",
            f"Responses applied: {values['responses_applied']}.",
            f"Coverage restored to the empirical floor: {values['coverage_restored_episodes']} episodes.", ""]
    lines += ["## Utility trade-offs", "",
        "| Scenario | Seed | A/B mean set size | A/B singleton rate | A/B benign review rate |",
        "|---|---:|---|---|---|"]
    for row in card["episodes"]:
        a, b = row["A"]["final_holdout"], row["B"]["final_holdout"]
        lines.append(f"| {row['scenario']} | {row['seed']} | {a['mean_set_size']:.3f} / {b['mean_set_size']:.3f} | "
            f"{_percent(a['singleton_rate'])} / {_percent(b['singleton_rate'])} | "
            f"{_percent(a['by_class']['0']['review_rate'])} / {_percent(b['by_class']['0']['review_rate'])} |")
    lines += ["", "## Limits", ""] + [f"- {value}" for value in card["limits"]]
    return "\n".join(lines) + "\n"


def write_scorecard(directory, comparisons, metadata):
    directory = Path(directory)
    card = scorecard(comparisons, metadata)
    write_new_json(directory / "scorecard.json", card)
    with new_output(directory / "scorecard.md") as handle:
        handle.write(markdown(card))
    with new_output(directory / "scorecard.csv") as handle:
        fields = ["scenario", "seed", "first_alarm_index", "detection_latency_steps", "negative_control_alarm", "pre_shift_alarm",
            "response_status", "response_applied", "coverage_restored", "restoration_status"]
        metrics = [f"{arm}_{name}" for arm in ("A", "B") for name in ("unsafe_coverage", "benign_coverage", "review_rate", "unsafe_allow_rate")]
        writer = csv.DictWriter(handle, fieldnames=fields + metrics)
        writer.writeheader()
        for row in comparisons:
            values = {key: row[key] for key in fields}
            for arm in ("A", "B"):
                final = row[arm]["final_holdout"]
                values.update({f"{arm}_unsafe_coverage": final["by_class"]["1"]["coverage"],
                    f"{arm}_benign_coverage": final["by_class"]["0"]["coverage"],
                    f"{arm}_review_rate": final["review_rate"], f"{arm}_unsafe_allow_rate": final["unsafe_label_allow_rate"]})
            writer.writerow(values)
    return card


def main(argv=None):
    parser = argparse.ArgumentParser(description="Regenerate scorecard from frozen paired summaries")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    source = strict_json((args.results / "paired-summaries.json").read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_scorecard(args.output_dir, source["comparisons"], source["metadata"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
