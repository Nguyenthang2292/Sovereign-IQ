import sys
from unittest.mock import MagicMock, patch

import pytest

# Define the module name to reload
MODULE_NAME = 'modules.auto_trade.gui.components.empty_state'

@pytest.fixture
def mock_ctk():
    """
    Patches customtkinter and returns the mock.
    """
    # Create a mock module for customtkinter
    mock_module = MagicMock()

    # Define a dummy base class for CTkFrame that absorbs __init__ arguments
    # preventing them from being passed to MagicMock.__init__ as spec
    class MockCTkFrame:
        def __init__(self, *args, **kwargs):
            pass

        def pack(self, *args, **kwargs):
            pass

    # Setup specific mocks
    mock_module.CTkFrame = MockCTkFrame

    # We mock CTkLabel and CTkButton to return mock instances when instantiated
    mock_module.CTkLabel = MagicMock()
    mock_module.CTkButton = MagicMock()

    # Patch sys.modules to inject our mock
    with patch.dict(sys.modules, {'customtkinter': mock_module}):
        yield mock_module

@pytest.fixture
def empty_state_cls(mock_ctk):
    """
    Imports the EmptyState class after patching customtkinter.
    Forces a reload if already imported to ensure it picks up the mock.
    """
    if MODULE_NAME in sys.modules:
        del sys.modules[MODULE_NAME]

    from modules.auto_trade.gui.components.empty_state import EmptyState
    return EmptyState

def test_empty_state_structure(empty_state_cls, mock_ctk):
    """Test basic structure with icon and message only."""
    # Reset mocks to clear any calls
    mock_ctk.CTkLabel.reset_mock()
    mock_ctk.CTkButton.reset_mock()

    parent = MagicMock()
    # Instantiate EmptyState
    # Since EmptyState inherits from CTkFrame (which is mocked), this should work
    empty_state = empty_state_cls(
        parent,
        icon="📊",
        message="No Data"
    )

    # Verify Labels created
    # We expect 2 labels: icon and message
    assert mock_ctk.CTkLabel.call_count == 2, "Expected icon + message labels"

    # Check arguments
    calls = mock_ctk.CTkLabel.call_args_list
    texts = [c.kwargs.get('text') for c in calls]
    assert "📊" in texts, "Expected icon text in labels"
    assert "No Data" in texts, "Expected message text in labels"

    # Verify Button NOT created
    assert mock_ctk.CTkButton.call_count == 0, "Button should not be created without action"

def test_empty_state_with_hint(empty_state_cls, mock_ctk):
    """Test EmptyState with a hint text."""
    mock_ctk.CTkLabel.reset_mock()

    parent = MagicMock()
    empty_state = empty_state_cls(
        parent,
        icon="Icon",
        message="Message",
        hint="Hint Text"
    )

    # Expect 3 labels (icon, message, hint)
    assert mock_ctk.CTkLabel.call_count == 3, "Expected icon, message, and hint labels"

    calls = mock_ctk.CTkLabel.call_args_list
    texts = [c.kwargs.get('text') for c in calls]
    assert "Hint Text" in texts, "Expected hint text in labels"

def test_empty_state_with_action(empty_state_cls, mock_ctk):
    """Test EmptyState with an action button."""
    mock_ctk.CTkButton.reset_mock()

    parent = MagicMock()
    mock_callback = MagicMock()

    empty_state = empty_state_cls(
        parent,
        icon="Icon",
        message="Message",
        action_text="Retry",
        action_callback=mock_callback
    )

    # Expect Button created
    assert mock_ctk.CTkButton.call_count == 1, "Expected action button when callback is provided"

    # Check kwargs
    call_args = mock_ctk.CTkButton.call_args
    assert call_args.kwargs.get('text') == "Retry", "Expected action button text to be set"
    command = call_args.kwargs.get('command')
    assert command is not None, "Expected action button command to be set"

    # Simulate click
    command()
    mock_callback.assert_called_once()

def test_empty_state_no_action_if_no_callback(empty_state_cls, mock_ctk):
    """Test button not created if callback missing."""
    mock_ctk.CTkButton.reset_mock()

    parent = MagicMock()
    empty_state = empty_state_cls(
        parent,
        icon="Icon",
        message="Message",
        action_text="Retry",
        action_callback=None
    )

    assert mock_ctk.CTkButton.call_count == 0, "Button should not be created without callback"

def test_action_callback_error_handling(empty_state_cls, mock_ctk):
    """Test error handling in callback."""
    parent = MagicMock()
    mock_callback = MagicMock(side_effect=ValueError("Boom"))

    empty_state = empty_state_cls(
        parent,
        icon="Icon",
        message="Message",
        action_text="Retry",
        action_callback=mock_callback
    )

    # Get the command
    # mock_ctk.CTkButton is the mock class/constructor
    # .call_args gets the arguments of the last call (instantiation)
    command = mock_ctk.CTkButton.call_args.kwargs.get('command')

    # We need to patch logging where it is used
    # It is imported in the module, so we need to patch MODULE_NAME.logging
    with patch(f"{MODULE_NAME}.logging") as mock_logging:
        # Executing the command should catch the exception
        command()

        mock_callback.assert_called_once()
        mock_logging.error.assert_called_once()
        # Verify error message contains exception text
        assert "Boom" in str(mock_logging.error.call_args), "Expected error text to be logged"
