from __future__ import annotations

import argparse

from etl.pipeline import available_steps, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cricket-pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run a pipeline stage or a single step.")
    run.add_argument("--stage", choices=["bronze", "silver", "gold", "all"], default="all")
    run.add_argument("--step", choices=[step.name for step in available_steps()])
    run.add_argument("--sample", action="store_true", help="Use a small input subset.")
    run.add_argument("--no-write", action="store_true", help="Preview without writing tables.")
    run.add_argument("--preview", action="store_true", help="Print sample rows and schemas.")

    sub.add_parser("list-steps", help="List registered pipeline steps.")
    inspect = sub.add_parser("inspect-raw", help="Print raw Cricsheet JSON schema samples.")
    inspect.add_argument("--files", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "list-steps":
        for step in available_steps():
            print(f"{step.stage}: {step.name}")
        return
    if args.command == "inspect-raw":
        from etl.inspect_raw import run

        run(sample_files=args.files)
        return

    run_pipeline(
        stage=args.stage,
        step_name=args.step,
        sample_only=args.sample,
        write_out=not args.no_write,
        preview=args.preview or args.no_write,
    )


if __name__ == "__main__":
    main()
