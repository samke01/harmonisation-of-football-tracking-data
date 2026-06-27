"""Run the final harmonization evaluation suite.

Usage examples:
    python -m scripts.run_final_evaluation --task all
    python -m scripts.run_final_evaluation --task player-aggregate
    python -m scripts.run_final_evaluation --task kinematic-regression
"""

from __future__ import annotations

import argparse
import sys

from src.evaluation import run_harmonization_evaluation
from src.evaluation.downstream.experiments import (
    run_ball_status_learning_curves,
    run_ball_status_transfer,
    run_kinematic_regression_learning_curves,
    run_kinematic_regression_target_augmentation,
    run_kinematic_regression_transfer,
    run_pass_success_event_learning_curves,
    run_pass_success_event_target_augmentation,
    run_pass_success_event_transfer,
    run_pass_success_tracking_learning_curves,
    run_pass_success_tracking_target_augmentation,
    run_pass_success_tracking_transfer,
    run_player_aggregate_position_learning_curves,
    run_player_aggregate_position_target_augmentation,
    run_player_aggregate_position_transfer,
    run_position_label_audit,
)


FINAL_TASKS = [
    "diagnostics",
    "audit",
    "player-aggregate",
    "kinematic-regression",
    "ball-status",
    "pass-success-event",
    "pass-success-tracking",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=[*FINAL_TASKS, "all"],
        default="all",
        help="Which final evaluation task to run.",
    )
    parser.add_argument("--learning-curves", action="store_true")
    parser.add_argument("--transfer", action="store_true")
    parser.add_argument(
        "--target-augmentation",
        action="store_true",
        help=(
            "Run per-target augmentation evaluation: fixed target-source test "
            "set, four train regimes (target_only, other_only, merged_same_n, "
            "target_plus_other_2n)."
        ),
    )
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    run_both = not (args.learning_curves or args.transfer or args.target_augmentation)
    tasks = FINAL_TASKS if args.task == "all" else [args.task]

    for task in tasks:
        if task == "diagnostics":
            print("Running domain-shift diagnostics...")
            run_harmonization_evaluation()
        elif task == "audit":
            print("Running position-label audit...")
            run_position_label_audit()
        elif task == "player-aggregate":
            if args.learning_curves or run_both:
                print("Running player-aggregate position learning curves...")
                run_player_aggregate_position_learning_curves()
            if args.target_augmentation or run_both:
                print("Running player-aggregate position per-target augmentation...")
                run_player_aggregate_position_target_augmentation()
            if args.transfer or run_both:
                print("Running player-aggregate position transfer...")
                run_player_aggregate_position_transfer()
        elif task == "kinematic-regression":
            if args.learning_curves or run_both:
                print("Running kinematic regression learning curves...")
                run_kinematic_regression_learning_curves()
            if args.target_augmentation or run_both:
                print("Running kinematic regression per-target augmentation...")
                run_kinematic_regression_target_augmentation()
            if args.transfer or run_both:
                print("Running kinematic regression transfer...")
                run_kinematic_regression_transfer()
        elif task == "ball-status":
            if args.learning_curves or run_both:
                print("Running ball-status learning curves...")
                run_ball_status_learning_curves()
            if args.transfer or run_both:
                print("Running ball-status transfer...")
                run_ball_status_transfer()
        elif task == "pass-success-event":
            if args.learning_curves or run_both:
                print("Running pass-success (event-only) learning curves...")
                run_pass_success_event_learning_curves()
            if args.target_augmentation or run_both:
                print("Running pass-success (event-only) per-target augmentation...")
                run_pass_success_event_target_augmentation()
            if args.transfer or run_both:
                print("Running pass-success (event-only) transfer...")
                run_pass_success_event_transfer()
        elif task == "pass-success-tracking":
            if args.learning_curves or run_both:
                print("Running pass-success (tracking-context) learning curves...")
                run_pass_success_tracking_learning_curves()
            if args.target_augmentation or run_both:
                print("Running pass-success (tracking-context) per-target augmentation...")
                run_pass_success_tracking_target_augmentation()
            if args.transfer or run_both:
                print("Running pass-success (tracking-context) transfer...")
                run_pass_success_tracking_transfer()


if __name__ == "__main__":
    main()
