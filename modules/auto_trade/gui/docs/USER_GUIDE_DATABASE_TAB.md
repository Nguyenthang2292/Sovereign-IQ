# Database Testing Panel User Guide

## Overview

The Database Testing Panel is a comprehensive tool integrated into the Auto-Trade Dashboard for managing, testing, and monitoring the trading database. It allows users to:

- Create test orders and signals to verify system behavior.
- Monitor active martingale chains.
- View and export database records.
- Perform maintenance tasks like backups and cleanups.
- Analyze trading statistics.

## Getting Started

1.  Launch the Auto-Trade Dashboard.
2.  Navigate to the **Database** tab.
3.  The panel is divided into two main areas:
    -   **Left Panel**: Testing controls and Data Viewer.
    -   **Right Panel**: Database Stats, Quick Actions, and Activity Logs.

## Orders Testing

Use this section to simulate order creation and analyze order data.

-   **Create Test Order**:
    1.  Enter the **Symbol** (e.g., BTCUSDT).
    2.  Select the **Side** (LONG/SHORT).
    3.  Click **Create Test Order**. This creates a programmatic order in the database.
-   **Query Open Positions**: Displays all currently open orders in the Data Viewer.
-   **Get Overall Stats**: Shows summary statistics like total PnL, win rate, etc.
-   **Get Daily Stats (30d)**: Displays a daily breakdown of trading activity for the last 30 days.

## Signals Testing

Use this section to verify signal processing.

-   **Create Test Signal**:
    1.  Enter the **Symbol**.
    2.  Enter **Confidence** (0.0 to 1.0).
    3.  Click **Create Test Signal**.
-   **Get Recent Signals**: Lists the latest signals generated or manually created.
-   **Signal Performance Stats**: Shows performance metrics based on signal confidence and outcomes.

## Martingale Testing

Monitor the status of recovery chains.

-   **Get Active Chains**: Lists all currently active martingale recovery sequences, showing the current step and accumulated PnL.
-   **Chain Statistics**: Provides a summary of total, active, and completed/recovered chains.

## Database Operations (Quick Actions)

Perform essential maintenance and inspection tasks.

-   **💾 Create Backup**: Creates a timestamped backup of the SQLite database in the `data/backups` directory.
-   **🔄 Run Migrations**: Checks for and applies any pending schema migrations.
-   **🧹 Cleanup Old Records**: Deletes orders, signals, and logs older than 90 days (requires confirmation).
-   **📤 Export to CSV**: Exports the currently selected table (in Data Viewer) to a CSV file.
-   **📋 View Audit Log**: Displays the most recent system audit logs for debugging.
-   **🔍 Check Integrity**: Runs a SQLite integrity check to ensure database health.

## Data Viewer

The Data Viewer allows you to explore the raw data in the database.

-   **Table Selector**: Choose between Orders, Signals, Martingale Chains, and Audit Log.
-   **Pagination**: Use **< Prev** and **Next >** buttons to navigate through large datasets (20 records per page).
-   **Refresh**: The viewer automatically refreshes when you perform queries or change tables.

## Troubleshooting

-   **"Migration manager not available"**: Ensure the database configuration paths are correct.
-   **Stats not updating**: Click any query button to force a refresh, or check the Activity Logs for errors.
-   **Backup failed**: Check file permissions for the `data/backups` directory.

## FAQ

**Q: Does "Create Test Order" execute a real trade on Binance?**
A: No. It only creates a record in the local database with `order_source='PROGRAMMATIC'` and `execution_mode='AUTO'`. It does **not** send an API request to the exchange.

**Q: Can I delete a specific order?**
A: Currently, only bulk cleanup of old records is supported via the GUI. Use a SQLite client for specific deletions.
