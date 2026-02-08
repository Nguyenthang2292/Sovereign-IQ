"""
Database Backup and Recovery System
=====================================

Handles automated backups, compression, and recovery for SQLite database.

Created: 2026-02-03
"""

import datetime
import gzip
import json
import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class BackupManager:
    """
    Manages database backups and recovery.
    """

    def __init__(
        self, db_path: str, backup_dir: str = "data/backups", max_backups: int = 30, compress: bool = True
    ) -> None:
        """
        Initialize backup manager.

        Args:
            db_path: Path to SQLite database file
            backup_dir: Directory to store backups
            max_backups: Maximum number of backups to keep
            compress: Whether to compress backups
        """
        self.db_path = db_path
        self.backup_dir = Path(backup_dir)
        self.max_backups = max_backups
        self.compress = compress

        # Create backup directory if it doesn't exist
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def create_backup(self, backup_name: Optional[str] = None, metadata: Optional[dict] = None) -> Optional[str]:
        """
        Create a database backup.

        Args:
            backup_name: Optional custom backup name (else auto-generated)
            metadata: Optional metadata to store with backup

        Returns:
            Path to backup file or None if failed
        """
        if not os.path.exists(self.db_path):
            logger.error(f"Database file not found: {self.db_path}")
            return None

        try:
            # Generate backup filename
            if not backup_name:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"auto_trade_backup_{timestamp}"

            if self.compress:
                backup_file = self.backup_dir / f"{backup_name}.db.gz"
            else:
                backup_file = self.backup_dir / f"{backup_name}.db"

            logger.info(f"Creating backup: {backup_file}")

            # Create backup
            if self.compress:
                # Compress while copying
                with open(self.db_path, "rb") as f_in:
                    with gzip.open(backup_file, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                # Direct copy
                shutil.copy2(self.db_path, backup_file)

            # Create metadata file
            if metadata:
                metadata_file = backup_file.with_suffix(backup_file.suffix + ".meta")
                metadata["backup_time"] = datetime.datetime.now().isoformat()
                metadata["original_db_path"] = self.db_path
                metadata["compressed"] = self.compress

                with open(metadata_file, "w") as f:
                    json.dump(metadata, f, indent=2)

            # Verify backup
            if self.verify_backup(str(backup_file)):
                logger.info(f"Backup created successfully: {backup_file}")

                # Cleanup old backups
                self.cleanup_old_backups()

                return str(backup_file)
            else:
                logger.error(f"Backup verification failed: {backup_file}")
                # Delete failed backup
                if backup_file.exists():
                    backup_file.unlink()
                return None

        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return None

    def verify_backup(self, backup_path: str) -> bool:
        """
        Verify that a backup file is valid.

        Args:
            backup_path: Path to backup file

        Returns:
            True if backup is valid
        """
        try:
            path = Path(backup_path)

            if not path.exists():
                return False

            # Check file size
            if path.stat().st_size == 0:
                logger.error("Backup file is empty")
                return False

            # For compressed files, try to decompress and check
            if path.suffix == ".gz":
                try:
                    with gzip.open(path, "rb") as f:
                        # Read first few bytes to verify it's valid gzip
                        f.read(100)
                    return True
                except Exception as e:
                    logger.error(f"Backup compression verification failed: {e}")
                    return False
            else:
                # For uncompressed, check if it's a valid SQLite file
                try:
                    import sqlite3

                    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master LIMIT 1")
                    conn.close()
                    return True
                except Exception as e:
                    logger.error(f"Backup SQLite verification failed: {e}")
                    return False

        except Exception as e:
            logger.error(f"Error verifying backup: {e}")
            return False

    def restore_backup(self, backup_path: str, target_path: Optional[str] = None) -> bool:
        """
        Restore database from backup.

        Args:
            backup_path: Path to backup file
            target_path: Optional target path (default: original db_path)

        Returns:
            True if restored successfully
        """
        if not target_path:
            target_path = self.db_path

        path = Path(backup_path)

        if not path.exists():
            logger.error(f"Backup file not found: {path}")
            return False

        try:
            # Create backup of current database before restore
            if os.path.exists(target_path):
                pre_restore_backup = self.create_backup(
                    backup_name=f"pre_restore_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    metadata={"restore_source": str(path)},
                )
                logger.info(f"Created pre-restore backup: {pre_restore_backup}")

            logger.info(f"Restoring from backup: {path}")

            # Restore
            if path.suffix == ".gz":
                # Decompress
                with gzip.open(path, "rb") as f_in:
                    with open(target_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                # Direct copy
                shutil.copy2(path, target_path)

            # Verify restored database
            import sqlite3

            conn = sqlite3.connect(target_path)
            cursor = conn.cursor()
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()
            conn.close()

            if result and result[0] == "ok":
                logger.info(f"Database restored successfully to {target_path}")
                return True
            else:
                logger.error(f"Restored database failed integrity check: {result}")
                return False

        except Exception as e:
            logger.error(f"Error restoring backup: {e}")
            return False

    def list_backups(self) -> List[dict]:
        """
        List all available backups.

        Returns:
            List of backup info dictionaries
        """
        backups = []

        # Find all backup files
        backup_files = sorted(self.backup_dir.glob("*.db*"), key=lambda p: p.stat().st_mtime, reverse=True)

        for backup_file in backup_files:
            # Skip metadata files
            if backup_file.suffix == ".meta":
                continue

            # Get file info
            stat = backup_file.stat()

            backup_info = {
                "filename": backup_file.name,
                "path": str(backup_file),
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "created": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "compressed": backup_file.suffix == ".gz",
            }

            # Load metadata if exists
            metadata_file = backup_file.with_suffix(backup_file.suffix + ".meta")
            if metadata_file.exists():
                try:
                    with open(metadata_file, "r") as f:
                        backup_info["metadata"] = json.load(f)
                except Exception:
                    pass

            backups.append(backup_info)

        return backups

    def cleanup_old_backups(self) -> int:
        """
        Remove old backups exceeding max_backups limit.

        Returns:
            Number of backups deleted
        """
        backups = self.list_backups()

        if len(backups) <= self.max_backups:
            return 0

        # Delete oldest backups
        to_delete = backups[self.max_backups :]
        deleted_count = 0

        for backup in to_delete:
            try:
                backup_path = Path(backup["path"])
                backup_path.unlink()

                # Delete metadata file if exists
                metadata_file = backup_path.with_suffix(backup_path.suffix + ".meta")
                if metadata_file.exists():
                    metadata_file.unlink()

                logger.info(f"Deleted old backup: {backup['filename']}")
                deleted_count += 1

            except Exception as e:
                logger.error(f"Error deleting backup {backup['filename']}: {e}")

        return deleted_count

    def get_latest_backup(self) -> Optional[dict]:
        """
        Get the most recent backup.

        Returns:
            Backup info dictionary or None
        """
        backups = self.list_backups()
        return backups[0] if backups else None

    def schedule_backup(self) -> bool:
        """
        Create a scheduled backup (called by scheduler).

        Returns:
            True if successful
        """
        # Create backup with current stats
        import sqlite3

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get basic stats
            cursor.execute("SELECT COUNT(*) FROM orders WHERE order_source='PROGRAMMATIC'")
            total_orders = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM signals")
            total_signals = cursor.fetchone()[0]

            conn.close()

            metadata = {"backup_type": "scheduled", "total_orders": total_orders, "total_signals": total_signals}

            backup_path = self.create_backup(metadata=metadata)

            if backup_path:
                logger.info(f"Scheduled backup created: {backup_path}")
                return True
            else:
                logger.error("Scheduled backup failed")
                return False

        except Exception as e:
            logger.error(f"Error in scheduled backup: {e}")
            return False

    def export_to_sql(self, output_path: str) -> bool:
        """
        Export database to SQL dump file.

        Args:
            output_path: Path for SQL dump file

        Returns:
            True if successful
        """
        try:
            import sqlite3

            conn = sqlite3.connect(self.db_path)

            with open(output_path, "w", encoding="utf-8") as f:
                for line in conn.iterdump():
                    f.write(f"{line}\n")

            conn.close()

            logger.info(f"Database exported to SQL: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting to SQL: {e}")
            return False

    def get_backup_stats(self) -> dict:
        """
        Get backup statistics.

        Returns:
            Dictionary with backup statistics
        """
        backups = self.list_backups()

        if not backups:
            return {"total_backups": 0, "total_size_mb": 0.0, "latest_backup": None, "oldest_backup": None}

        total_size = sum(b["size_bytes"] for b in backups)

        return {
            "total_backups": len(backups),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "latest_backup": backups[0]["created"],
            "oldest_backup": backups[-1]["created"],
            "compressed_count": sum(1 for b in backups if b["compressed"]),
        }


# ============================================================================
# BACKUP SCHEDULER
# ============================================================================


class BackupScheduler:
    """
    Handles scheduled automatic backups.
    """

    def __init__(self, backup_manager: BackupManager) -> None:
        """
        Initialize backup scheduler.

        Args:
            backup_manager: BackupManager instance
        """
        self.backup_manager = backup_manager
        self.last_backup_time: Optional[datetime.datetime] = None

    def should_backup(self, interval_hours: int = 24) -> bool:
        """
        Check if backup is needed based on interval.

        Args:
            interval_hours: Backup interval in hours

        Returns:
            True if backup is needed
        """
        if not self.last_backup_time:
            return True

        elapsed = datetime.datetime.now() - self.last_backup_time
        return elapsed.total_seconds() / 3600 >= interval_hours

    def run_if_needed(self, interval_hours: int = 24) -> bool:
        """
        Run backup if needed based on interval.

        Args:
            interval_hours: Backup interval in hours

        Returns:
            True if backup was created
        """
        if self.should_backup(interval_hours):
            if self.backup_manager.schedule_backup():
                self.last_backup_time = datetime.datetime.now()
                return True

        return False


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def create_backup(db_path: str, backup_dir: str = "data/backups", compress: bool = True) -> Optional[str]:
    """
    Quick function to create a database backup.

    Args:
        db_path: Path to database file
        backup_dir: Backup directory
        compress: Whether to compress

    Returns:
        Path to backup file or None
    """
    manager = BackupManager(db_path, backup_dir, compress=compress)
    return manager.create_backup()


def restore_latest_backup(db_path: str, backup_dir: str = "data/backups") -> bool:
    """
    Restore from the latest backup.

    Args:
        db_path: Path to database file
        backup_dir: Backup directory

    Returns:
        True if successful
    """
    manager = BackupManager(db_path, backup_dir)
    latest = manager.get_latest_backup()

    if latest:
        return manager.restore_backup(latest["path"], db_path)
    else:
        logger.error("No backups found")
        return False


def list_all_backups(backup_dir: str = "data/backups") -> List[dict]:
    """
    List all available backups.

    Args:
        backup_dir: Backup directory

    Returns:
        List of backup info dictionaries
    """
    # Use dummy db_path for listing
    manager = BackupManager("", backup_dir)
    return manager.list_backups()
