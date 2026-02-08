from __future__ import annotations

"""
Auto Trading System - Main Event Loop
======================================

Main orchestrator that integrates all modules:
- Signal generation (ATC + XGBoost pipeline)
- Order execution with risk management
- Position monitoring with break-even and Martingale
- Database persistence

Created: 2026-02-03
"""

import asyncio
import logging
import signal as signal_module
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import database components
from modules.auto_trade.auto_trade_config import AutoTradeConfig, load_config
from modules.auto_trade.database import (
    create_audit_log,
    create_database_backup,
    get_db_manager,
    get_open_positions,
    session_scope,
)

# Import order tagging
from modules.auto_trade.execution.order_tagging import OrderTagger, tag_programmatic_order

# Configure logging
log_dir = Path("data/logs")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_dir / "auto_trade.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


class AutoTradeSystem:
    """
    Main auto trading system orchestrator.

    Coordinates all modules in a continuous event loop:
    1. Check for open positions
    2. If no positions, scan for signals
    3. If signal found, execute order
    4. Monitor positions for break-even and Martingale
    """

    def __init__(self, config: Optional[AutoTradeConfig] = None):
        """
        Initialize auto trading system.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or load_config()
        self.running = False
        self.shutdown_requested = False

        # Module placeholders (to be implemented in respective phases)
        self.signal_scanner = None
        self.order_executor = None
        self.position_monitor = None

        # Statistics (start_time is datetime; counters are int)
        self.stats: Dict[str, Any] = {
            "loops_completed": 0,
            "signals_found": 0,
            "orders_executed": 0,
            "errors": 0,
            "start_time": None,
        }

        logger.info("AutoTradeSystem initialized")

    async def initialize(self) -> None:
        """Initialize all modules and database."""
        logger.info("=" * 60)
        logger.info("Initializing Auto Trading System...")
        logger.info("=" * 60)

        try:
            # Initialize database
            logger.info(f"Initializing database at {self.config.database.path}...")
            # Use get_db_manager to initialize singleton with config params
            get_db_manager(db_path=self.config.database.path, pool_size=self.config.database.pool_size, initialize=True)
            logger.info("[OK] Database initialized")

            # Create initial backup
            logger.info("Creating initial database backup...")
            backup_path = create_database_backup(compress=self.config.database.backup_compress)
            logger.info(f"[OK] Backup created: {backup_path}")

            # Log system start
            with session_scope() as session:
                audit_log = create_audit_log(
                    session,
                    event_type="SYSTEM",
                    event_category="STARTUP",
                    event_summary="Auto trading system started",
                    severity="INFO",
                )
                audit_log.set_event_data({"config": self.config.to_dict()})

            # TODO: Initialize signal scanner (Phase 2)
            # self.signal_scanner = SignalScanner(...)

            # TODO: Initialize order executor (Phase 3)
            # self.order_executor = OrderExecutor(...)

            # TODO: Initialize position monitor (Phase 4)
            # self.position_monitor = PositionMonitor(...)

            logger.info("[OK] All modules initialized")
            logger.info("=" * 60)

            self.stats["start_time"] = datetime.now(timezone.utc)

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise

    async def main_loop(self) -> None:
        """
        Main event loop.

        Continuously:
        1. Check for open positions
        2. Scan for new signals if slots available
        3. Execute orders for valid signals
        4. Monitor existing positions
        """
        logger.info("Starting main event loop...")
        self.running = True
        last_backup = time.time()

        while self.running and not self.shutdown_requested:
            try:
                loop_start = time.time()

                # Step 1: Check open positions
                with session_scope() as session:
                    open_positions = get_open_positions(session)
                    position_count = len(open_positions)

                logger.info(f"Open positions: {position_count}/{self.config.risk.max_open_positions}")

                # Step 2: Monitor existing positions
                if position_count > 0:
                    logger.info("Monitoring existing positions...")
                    await self._monitor_positions(open_positions)

                # Step 3: Scan for new signals if we have capacity
                if position_count < self.config.risk.max_open_positions:
                    logger.info("Scanning for new trading signals...")
                    signals = await self._scan_for_signals()

                    if signals:
                        logger.info(f"Found {len(signals)} potential signals")
                        self.stats["signals_found"] = (self.stats.get("signals_found") or 0) + len(signals)

                        # Execute orders for valid signals
                        for signal in signals:
                            if position_count >= self.config.risk.max_open_positions:
                                logger.info("Max positions reached, skipping remaining signals")
                                break

                            success = await self._execute_signal(signal)
                            if success:
                                position_count += 1
                                self.stats["orders_executed"] = (self.stats.get("orders_executed") or 0) + 1

                # Step 4: Periodic database backup
                if time.time() - last_backup > self.config.database.backup_interval:
                    logger.info("Creating periodic database backup...")
                    create_database_backup(compress=self.config.database.backup_compress)
                    last_backup = time.time()

                # Update stats
                self.stats["loops_completed"] = (self.stats.get("loops_completed") or 0) + 1

                # Log loop completion
                loop_time = time.time() - loop_start
                logger.info(f"Loop completed in {loop_time:.2f}s")
                logger.info(f"Stats: {self.stats}")

                # Sleep until next scan interval
                await self._sleep_until_next_scan()

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt detected")
                self.shutdown_requested = True
                break

            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                self.stats["errors"] = (self.stats.get("errors") or 0) + 1

                # Log error to database
                try:
                    with session_scope() as session:
                        create_audit_log(
                            session,
                            event_type="ERROR",
                            event_category="MAIN_LOOP",
                            event_summary=str(e),
                            severity="ERROR",
                        )
                except Exception as db_error:
                    logger.error(f"Failed to log error to database: {db_error}")

                # Wait before retrying
                await asyncio.sleep(10)

        logger.info("Main event loop stopped")

    async def _monitor_positions(self, positions: list) -> None:
        """
        Monitor open positions for break-even and Martingale.

        Args:
            positions: List of open position records
        """
        if not self.position_monitor:
            logger.debug("Position monitor not initialized (Phase 4), skipping...")
            return

        # TODO: Implement when Phase 4 is ready
        # for position in positions:
        #     await self.position_monitor.check_breakeven(position)
        #     await self.position_monitor.check_martingale(position)

        logger.debug(f"Monitored {len(positions)} positions")

    async def _scan_for_signals(self) -> list:
        """
        Scan market for trading signals.

        Returns:
            List of signal dictionaries
        """
        if not self.signal_scanner:
            logger.debug("Signal scanner not initialized (Phase 2), returning empty...")
            return []

        # TODO: Implement when Phase 2 is ready
        # signals = await self.signal_scanner.scan()
        # return signals

        # Placeholder
        return []

    async def _execute_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Execute order for a trading signal.

        Args:
            signal: Signal dictionary

        Returns:
            True if order executed successfully
        """
        if not self.order_executor:
            logger.debug("Order executor not initialized (Phase 3), skipping...")
            return False

        try:
            # Generate signal correlation ID
            signal_id = OrderTagger.generate_signal_correlation_id(signal["symbol"], signal["signal_type"])

            # Tag order
            order_metadata = tag_programmatic_order(symbol=signal["symbol"], signal_id=signal_id)

            logger.info(f"Executing order for {signal['symbol']} {signal['signal_type']}")
            logger.info(f"Client Order ID: {order_metadata['client_order_id']}")

            # TODO: Implement when Phase 3 is ready
            # success = await self.order_executor.execute(signal, order_metadata)
            # return success

            # Placeholder
            return False

        except Exception as e:
            logger.error(f"Failed to execute signal: {e}", exc_info=True)
            return False

    async def _sleep_until_next_scan(self) -> None:
        """Sleep until next scan interval."""
        interval = self.config.scanning.scan_interval
        logger.info(f"Sleeping for {interval}s until next scan...")
        await asyncio.sleep(interval)

    async def shutdown(self) -> None:
        """Graceful shutdown of the system."""
        logger.info("=" * 60)
        logger.info("Shutting down Auto Trading System...")
        logger.info("=" * 60)

        self.running = False

        # Create final backup
        logger.info("Creating final database backup...")
        try:
            backup_path = create_database_backup(compress=self.config.database.backup_compress)
            logger.info(f"[OK] Final backup created: {backup_path}")
        except Exception as e:
            logger.error(f"Failed to create final backup: {e}")

        # Log shutdown
        try:
            with session_scope() as session:
                audit_log = create_audit_log(
                    session,
                    event_type="SYSTEM",
                    event_category="SHUTDOWN",
                    event_summary="Auto trading system shutdown",
                    severity="INFO",
                )
                audit_log.set_event_data({"stats": self.stats})
        except Exception as e:
            logger.error(f"Failed to log shutdown: {e}")

        # Print final stats
        start_time = self.stats.get("start_time")
        if start_time is not None:
            runtime = datetime.now(timezone.utc) - start_time
            logger.info("Final Statistics:")
            logger.info(f"  Runtime: {runtime}")
            logger.info(f"  Loops completed: {self.stats['loops_completed']}")
            logger.info(f"  Signals found: {self.stats['signals_found']}")
            logger.info(f"  Orders executed: {self.stats['orders_executed']}")
            logger.info(f"  Errors: {self.stats['errors']}")
        else:
            logger.info("System never fully initialized")

        logger.info("=" * 60)
        logger.info("Shutdown complete")
        logger.info("=" * 60)

    def signal_handler(self, signum: int, frame: Any) -> None:
        """Handle system signals for graceful shutdown."""
        logger.info(f"Received signal {signum}")
        self.shutdown_requested = True


async def main():
    """Main entry point."""
    # Create system instance
    system = AutoTradeSystem()

    # Setup signal handlers
    signal_module.signal(signal_module.SIGINT, system.signal_handler)
    signal_module.signal(signal_module.SIGTERM, system.signal_handler)

    try:
        # Initialize
        await system.initialize()

        # Run main loop
        await system.main_loop()

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)

    finally:
        # Shutdown
        await system.shutdown()


if __name__ == "__main__":
    # Run async main
    asyncio.run(main())
