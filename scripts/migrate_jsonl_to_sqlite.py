#!/usr/bin/env python3
"""Migrate JSONL signal history to SQLite database."""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modules.auto_trade.core.persistence_sqlite import SignalPersistenceSQLite
from modules.auto_trade.core.signal_selector import FinalSignal
from modules.common.ui.logging import log_error, log_info


def migrate_jsonl_to_sqlite(
    jsonl_dir: str = "data/signals", db_path: str = "data/signals/signals.db", dry_run: bool = False
) -> Dict[str, Any]:
    """
    Migrate JSONL files to SQLite database.

    Args:
        jsonl_dir: Directory containing JSONL files
        db_path: Target SQLite database path
        dry_run: If True, only count records without writing

    Returns:
        Migration statistics
    """
    stats = {"files_processed": 0, "records_migrated": 0, "records_failed": 0, "errors": []}

    jsonl_files = sorted(Path(jsonl_dir).glob("signal_history*.jsonl"))

    if not jsonl_files:
        log_error(f"No JSONL files found in {jsonl_dir}")
        return stats

    log_info(f"Found {len(jsonl_files)} JSONL files to migrate")

    if not dry_run:
        persistence = SignalPersistenceSQLite(db_path=db_path)

    for filepath in jsonl_files:
        stats["files_processed"] += 1
        log_info(f"Processing {filepath.name}...")

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue

                    try:
                        record = json.loads(line)

                        if dry_run:
                            stats["records_migrated"] += 1
                            continue

                        # Convert to FinalSignal
                        timestamp = datetime.fromisoformat(record["timestamp"]).timestamp()
                        signal = FinalSignal(
                            symbol=record["symbol"],
                            signal_type=record["type"],
                            confidence=record.get("confidence", 0.0),
                            entry_price=record["entry"],
                            stop_loss=record.get("stop_loss"),
                            take_profit=record.get("take_profit"),
                            sources=record.get("sources", []),
                            timestamp=timestamp,
                        )

                        signal_id = persistence.save_signal(signal)
                        if signal_id:
                            stats["records_migrated"] += 1
                        else:
                            stats["records_failed"] += 1

                    except Exception as e:
                        stats["records_failed"] += 1
                        error_msg = f"{filepath.name}:{line_num} - {str(e)}"
                        stats["errors"].append(error_msg)
                        if len(stats["errors"]) <= 10:
                            log_error(f"Failed to migrate record: {error_msg}")

        except Exception as e:
            error_msg = f"Failed to read {filepath.name}: {e}"
            stats["errors"].append(error_msg)
            log_error(error_msg)

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Migrate JSONL signal history to SQLite")
    parser.add_argument("--jsonl-dir", default="data/signals", help="Directory containing JSONL files")
    parser.add_argument("--db-path", default="data/signals/signals.db", help="SQLite database path")
    parser.add_argument("--dry-run", action="store_true", help="Only count records without writing")

    args = parser.parse_args()

    if args.dry_run:
        log_info("DRY RUN MODE - No data will be written")

    stats = migrate_jsonl_to_sqlite(jsonl_dir=args.jsonl_dir, db_path=args.db_path, dry_run=args.dry_run)

    print("\n" + "=" * 60)
    print("Migration Summary")
    print("=" * 60)
    print(f"Files processed: {stats['files_processed']}")
    print(f"Records migrated: {stats['records_migrated']}")
    print(f"Records failed: {stats['records_failed']}")

    if stats["errors"]:
        print(f"\nErrors ({len(stats['errors'])} total):")
        for error in stats["errors"][:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(stats["errors"]) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more errors")

    print("\nMigration completed!")
