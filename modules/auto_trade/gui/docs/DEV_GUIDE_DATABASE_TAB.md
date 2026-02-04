# Database Testing Panel Developer Guide

## Architecture

The Database Testing Panel is implemented as a standalone component (`DatabasePanel`) in the Auto-Trade GUI. It follows the existing component structure using `customtkinter`.

### Component Structure

-   **Class**: `DatabasePanel` (inherits from `ctk.CTkFrame`)
-   **Location**: `modules/auto_trade/gui/components/database_panel.py`
-   **Layout**:
    -   `left_panel`: Contains testing inputs and the data viewer.
    -   `right_panel`: Contains stats, quick actions, and logs.

### Database Integration

The panel interacts with the database via the `modules.auto_trade.database` package. It uses the `session_scope()` context manager for all database operations to ensure thread safety and proper resource management.

**Key Imports:**
```python
from modules.auto_trade.database import (
    session_scope, create_order, save_signal, get_open_positions, ...
)
from modules.auto_trade.database.models import Order, Signal, MartingaleChain
```

## Extension Points

### Adding New Tests

To add a new testing section (e.g., for a new strategy component):

1.  Create a new method `_create_new_section(self, parent)` in `DatabasePanel`.
2.  Call this method in `_create_layout`.
3.  Implement the logic methods (e.g., `_create_test_entity`).

### Adding New Actions

To add a new quick action:

1.  Add a tuple `("Label", self._method)` to the `actions` list in `_create_actions_section`.
2.  Implement the `_method` to perform the action.

## Testing Strategy

### Unit Tests

Unit tests are located in `tests/auto_trade/gui/test_database_panel.py`.
They use `unittest.mock` to mock the database and UI components.

**Running Tests:**
```bash
python -m unittest tests/auto_trade/gui/test_database_panel.py
```

### Manual Testing

Refer to `TASK_DATABASE_TAB_IMPLEMENTATION.md` (Task 8.3) for a comprehensive manual testing checklist.

## Common Issues

-   **Database Locks**: SQLite restricts concurrent writes. Always use `session_scope()` which handles commits and rollbacks.
-   **UI Freezing**: Long-running database operations (like huge exports) run on the main thread currently. For production use, consider moving these to a background thread using `threading`.
-   **Circular Imports**: Avoid importing `DatabasePanel` inside `main_window.py` at the top level if `main_window` is imported by components. Use local imports inside methods.
