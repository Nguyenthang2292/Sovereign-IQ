import sys
import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock, patch

import customtkinter as ctk

# Add project root to path
# File is at: modules/auto_trade/test_phase4_gui_integration.py
# Root is at: ../../
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from modules.auto_trade.gui.components.position_actions import PositionActions


class TestPhase4Integration(unittest.TestCase):
    root: ClassVar[ctk.CTk]

    @classmethod
    def setUpClass(cls):
        # Create a dummy root window for CTk components
        cls.root = ctk.CTk()
        # Hide window
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

    def setUp(self):
        self.mock_position = {
            "symbol": "BTC/USDT",
            "side": "LONG",
            "size": 0.1,
            "entry_price": 50000.0,
            "current_price": 51000.0,
            "liquidation_price": 40000.0,
            "unrealized_pnl": 100.0,
            "leverage": 10,
            "take_profit": 55000.0,
            "stop_loss": 48000.0,
            "id": "12345",
        }
        self.mock_callback = MagicMock()

    def test_position_actions_init(self):
        """Test PositionActions initialization"""
        actions = PositionActions(self.root, self.mock_position, self.mock_callback)
        self.assertIsInstance(actions, ctk.CTkFrame)
        # Check if sections exist
        self.assertTrue(hasattr(actions, "_create_close_section"))
        self.assertTrue(hasattr(actions, "_create_partial_close_section"))
        self.assertTrue(hasattr(actions, "_create_modify_tp_sl_section"))

    @patch("modules.auto_trade.gui.components.position_actions.messagebox.askyesno")
    def test_close_position_market(self, mock_askyesno):
        """Test Close Position (Market) action"""
        mock_askyesno.return_value = True  # Auto confirm

        actions = PositionActions(self.root, self.mock_position, self.mock_callback)

        # Simulate market close selection
        actions.close_type_var.set("market")

        # Trigger confirm
        actions._confirm_close_position()

        # Verify callback called with correct params
        self.mock_callback.assert_called_once()
        call_args = self.mock_callback.call_args[0][0]
        self.assertEqual(call_args["action"], "close_position")
        self.assertEqual(call_args["symbol"], "BTC/USDT")
        self.assertEqual(call_args["type"], "market")
        self.assertEqual(call_args["size"], 0.1)

    @patch("modules.auto_trade.gui.components.position_actions.messagebox.askyesno")
    def test_partial_close(self, mock_askyesno):
        """Test Partial Close action"""
        mock_askyesno.return_value = True

        actions = PositionActions(self.root, self.mock_position, self.mock_callback)

        # Set partial pct to 50%
        actions.partial_pct_var.set("50")

        # Trigger confirm
        actions._confirm_partial_close()

        # Verify callback
        self.mock_callback.assert_called_once()
        call_args = self.mock_callback.call_args[0][0]
        self.assertEqual(call_args["action"], "partial_close")
        self.assertEqual(call_args["percentage"], 50.0)
        self.assertAlmostEqual(call_args["size"], 0.05)  # 50% of 0.1

    @patch("modules.auto_trade.gui.components.position_actions.messagebox.askyesno")
    def test_modify_tp_sl(self, mock_askyesno):
        """Test Modify TP/SL action"""
        mock_askyesno.return_value = True

        actions = PositionActions(self.root, self.mock_position, self.mock_callback)

        # Set new TP/SL
        actions.tp_entry.insert(0, "60000")
        actions.sl_entry.insert(0, "49000")

        # Trigger confirm
        actions._confirm_modify_tp_sl()

        # Verify callback
        self.mock_callback.assert_called_once()
        call_args = self.mock_callback.call_args[0][0]
        self.assertEqual(call_args["action"], "modify_tp_sl")
        self.assertEqual(call_args["take_profit"], 60000.0)
        self.assertEqual(call_args["stop_loss"], 49000.0)

    @patch("modules.auto_trade.gui.components.position_actions.messagebox.askyesno")
    def test_add_margin(self, mock_askyesno):
        """Test Add Margin action"""
        mock_askyesno.return_value = True

        actions = PositionActions(self.root, self.mock_position, self.mock_callback)

        # Set margin amount
        actions.margin_entry.insert(0, "100")

        # Trigger confirm
        actions._confirm_add_margin()

        # Verify callback
        self.mock_callback.assert_called_once()
        call_args = self.mock_callback.call_args[0][0]
        self.assertEqual(call_args["action"], "add_margin")
        self.assertEqual(call_args["amount"], 100.0)

    def test_validation_logic(self):
        """Test local validation logic"""
        actions = PositionActions(self.root, self.mock_position, self.mock_callback)

        # Test TP below entry for LONG (Invalid)
        _ = actions._validate_tp_sl(tp=40000.0, sl=48000.0)
        # Note: _validate_tp_sl shows error dialog, checking logic directly
        # Since it calls messagebox.showerror, we should mock it,
        # but for simplicity we rely on the internal logic returning False

        with patch("modules.auto_trade.gui.components.position_actions.messagebox.showerror") as mock_error:
            self.assertFalse(actions._validate_tp_sl(tp=40000.0, sl=48000.0))
            mock_error.assert_called()

        # Test valid TP/SL
        with patch("modules.auto_trade.gui.components.position_actions.messagebox.showerror") as mock_error:
            self.assertTrue(actions._validate_tp_sl(tp=60000.0, sl=48000.0))
            mock_error.assert_not_called()


if __name__ == "__main__":
    unittest.main()
