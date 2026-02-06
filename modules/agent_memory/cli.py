"""CLI: python -m modules.agent_memory recall | store [summary]."""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Agent memory: recall (write context file) or store (add workflow to MemOS)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("recall", help="Search MemOS and write .cursor/agent_memory_context.md")

    store_p = sub.add_parser("store", help="Store a workflow summary in MemOS")
    store_p.add_argument(
        "summary",
        nargs="*",
        default=[],
        help="Summary text (or run without args to store last git commit)",
    )

    args = parser.parse_args()

    if args.command == "recall":
        from modules.agent_memory.recall import run_recall

        ok = run_recall()
        return 0 if ok else 1

    if args.command == "store":
        from modules.agent_memory.store import commit_summary, store_summary

        if args.summary:
            text = " ".join(args.summary)
            ok = store_summary(text)
        else:
            ok = commit_summary()
        return 0 if ok else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
