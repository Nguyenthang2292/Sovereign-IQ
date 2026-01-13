"""Tests for cli/display.py.

Tests cover:
- display_config function
- display_voting_metadata function
- Edge cases and different modes
"""

import re
from unittest.mock import patch

import pandas as pd

from cli.display import (
    display_config,
    display_voting_metadata,
)


class TestDisplayConfig:
    """Test display_config function."""

    @patch("cli.display.display_configuration")
    @patch("cli.display.log_progress")
    @patch("cli.display.log_data")
    def test_display_config_basic(self, mock_log_data, mock_log_progress, mock_display_config):
        """Test basic display_config call."""

        class Args:
            timeframe = "1h"
            limit = 500
            min_signal = 0.01
            max_symbols = None
            enable_spc = False

        def get_oscillator_params():
            return {"max_workers": 10, "strategies": [5, 6, 7]}

        display_config(selected_timeframe="1h", args=Args(), get_oscillator_params=get_oscillator_params, mode="hybrid")

        # Verify display_configuration was called
        mock_display_config.assert_called_once()
        call_args = mock_display_config.call_args
        assert call_args.kwargs["timeframe"] == "1h"
        assert call_args.kwargs["limit"] == 500

    @patch("cli.display.display_configuration")
    @patch("cli.display.log_progress")
    @patch("cli.display.log_data")
    def test_display_config_with_spc(self, mock_log_data, mock_log_progress, mock_display_config):
        """Test display_config with SPC enabled."""

        class Args:
            timeframe = "1h"
            limit = 500
            min_signal = 0.01
            max_symbols = None
            enable_spc = True

        def get_oscillator_params():
            return {"max_workers": 10, "strategies": [5, 6, 7]}

        def get_spc_params():
            return {"k": 2, "lookback": 100, "p_low": 20.0, "p_high": 80.0}

        display_config(
            selected_timeframe="1h",
            args=Args(),
            get_oscillator_params=get_oscillator_params,
            get_spc_params=get_spc_params,
            mode="hybrid",
        )

        # Verify SPC config was logged
        # Collect all call arguments as strings for searching
        call_strings = [str(call.args[0]) if call.args else str(call) for call in mock_log_data.call_args_list]
        combined_call_string = " ".join(call_strings)

        # Verify each SPC parameter appears in the logged calls with specific patterns
        # K parameter: require "K:" followed by "2" (word boundary to avoid matching "20", "200", etc.)
        assert re.search(r"K:\s*2\b", combined_call_string), "SPC parameter 'K: 2' not found in log_data calls"
        # Lookback parameter: require "Lookback:" followed by "100" (word boundary to avoid matching "1000", etc.)
        assert re.search(r"Lookback:\s*100\b", combined_call_string), (
            "SPC parameter 'Lookback: 100' not found in log_data calls"
        )
        # p_low parameter: require "Percentiles:" followed by "20" or "20.0" (with optional %)
        assert re.search(r"Percentiles:\s*20\.?0?\s*%", combined_call_string), (
            "SPC parameter 'Percentiles: ... 20.0%' (p_low) not found in log_data calls"
        )
        # p_high parameter: require "Percentiles:" with "80" or "80.0" after dash (with optional %)
        assert re.search(r"Percentiles:.*?-\s*80\.?0?\s*%", combined_call_string), (
            "SPC parameter 'Percentiles: ... - 80.0%' (p_high) not found in log_data calls"
        )

    @patch("cli.display.display_configuration")
    @patch("cli.display.log_progress")
    @patch("cli.display.log_data")
    def test_display_config_with_xgboost(self, mock_log_data, mock_log_progress, mock_display_config):
        """Test display_config with XGBoost enabled."""

        class Args:
            timeframe = "1h"
            limit = 500
            min_signal = 0.01
            max_symbols = None
            enable_spc = False
            enable_xgboost = True

        def get_oscillator_params():
            return {"max_workers": 10, "strategies": [5, 6, 7]}

        display_config(selected_timeframe="1h", args=Args(), get_oscillator_params=get_oscillator_params, mode="hybrid")

        # Verify XGBoost config was logged - check that message contains "xgboost"
        xgboost_found = False

        # Check mock_log_data calls
        for call in mock_log_data.mock_calls:
            # Check positional arguments
            for arg in call.args:
                if isinstance(arg, str) and "xgboost" in arg.lower():
                    xgboost_found = True
                    break
            # Check keyword arguments
            for key, value in call.kwargs.items():
                if isinstance(value, str) and "xgboost" in value.lower():
                    xgboost_found = True
                    break
            if xgboost_found:
                break

        # Check mock_log_progress calls if not found yet
        if not xgboost_found:
            for call in mock_log_progress.mock_calls:
                # Check positional arguments
                for arg in call.args:
                    if isinstance(arg, str) and "xgboost" in arg.lower():
                        xgboost_found = True
                        break
                # Check keyword arguments
                for key, value in call.kwargs.items():
                    if isinstance(value, str) and "xgboost" in value.lower():
                        xgboost_found = True
                        break
                if xgboost_found:
                    break

        assert xgboost_found, "XGBoost indicator not found in any log messages"

    @patch("cli.display.display_configuration")
    @patch("cli.display.log_progress")
    @patch("cli.display.log_data")
    def test_display_config_with_hmm(self, mock_log_data, mock_log_progress, mock_display_config):
        """Test display_config with HMM enabled."""

        class Args:
            timeframe = "1h"
            limit = 500
            min_signal = 0.01
            max_symbols = None
            enable_spc = False
            enable_hmm = True
            hmm_window_size = 100
            hmm_window_kama = 50
            hmm_fast_kama = 2
            hmm_slow_kama = 30
            hmm_orders_argrelextrema = 3
            hmm_strict_mode = True

        def get_oscillator_params():
            return {"max_workers": 10, "strategies": [5, 6, 7]}

        display_config(selected_timeframe="1h", args=Args(), get_oscillator_params=get_oscillator_params, mode="hybrid")

        # Verify HMM config was logged with expected parameters
        assert mock_log_data.called or mock_log_progress.called
        logged_messages = [str(call) for call in mock_log_data.call_args_list + mock_log_progress.call_args_list]
        assert any("hmm" in msg.lower() for msg in logged_messages), "HMM should be mentioned in logs"

    @patch("cli.display.display_configuration")
    @patch("cli.display.log_progress")
    @patch("cli.display.log_data")
    def test_display_config_decision_matrix_hybrid(self, mock_log_data, mock_log_progress, mock_display_config):
        """Test display_config with Decision Matrix in hybrid mode."""

        class Args:
            timeframe = "1h"
            limit = 500
            min_signal = 0.01
            max_symbols = None
            enable_spc = False
            use_decision_matrix = True
            voting_threshold = 0.5
            min_votes = 2

        def get_oscillator_params():
            return {"max_workers": 10, "strategies": [5, 6, 7]}

        display_config(selected_timeframe="1h", args=Args(), get_oscillator_params=get_oscillator_params, mode="hybrid")

        # Verify Decision Matrix config was logged with expected parameters
        assert mock_log_data.called or mock_log_progress.called
        logged_messages = [str(call) for call in mock_log_data.call_args_list + mock_log_progress.call_args_list]
        assert any("decision" in msg.lower() or "voting" in msg.lower() for msg in logged_messages), (
            "Decision Matrix/voting should be mentioned in logs"
        )

    @patch("cli.display.display_configuration")
    @patch("cli.display.log_progress")
    @patch("cli.display.log_data")
    def test_display_config_decision_matrix_voting(self, mock_log_data, mock_log_progress, mock_display_config):
        """Test display_config with Decision Matrix in voting mode."""

        class Args:
            timeframe = "1h"
            limit = 500
            min_signal = 0.01
            max_symbols = None
            enable_spc = False
            voting_threshold = 0.5
            min_votes = 2

        def get_oscillator_params():
            return {"max_workers": 10, "strategies": [5, 6, 7]}

        display_config(selected_timeframe="1h", args=Args(), get_oscillator_params=get_oscillator_params, mode="voting")

        # Verify voting mode config was logged
        assert mock_log_data.called or mock_log_progress.called
        logged_messages = [
            str(call.args[0]) if call.args else str(call)
            for call in mock_log_data.call_args_list + mock_log_progress.call_args_list
        ]
        assert any("voting" in msg.lower() for msg in logged_messages), "Voting mode should be mentioned"


class TestDisplayVotingMetadata:
    """Test display_voting_metadata function."""

    @patch("cli.display.log_progress")
    @patch("cli.display.log_data")
    def test_display_voting_metadata_empty_dataframe(self, mock_log_data, mock_log_progress):
        """Test display_voting_metadata with empty DataFrame."""
        df = pd.DataFrame()

        display_voting_metadata(df, signal_type="LONG")

        # Should not log anything for empty DataFrame
        assert not mock_log_data.called

    @patch("cli.display.log_progress")
    @patch("cli.display.log_data")
    def test_display_voting_metadata_long_signals(self, mock_log_data, mock_log_progress):
        """Test display_voting_metadata for LONG signals with detailed assertions."""
        df = pd.DataFrame(
            {
                "symbol": ["BTC/USDT", "ETH/USDT"],
                "weighted_score": [0.75, 0.65],
                "voting_breakdown": [
                    {
                        "spc": {"vote": 1, "weight": 0.3, "contribution": 30.0},
                        "xgboost": {"vote": 1, "weight": 0.2, "contribution": 20.0},
                    },
                    {
                        "spc": {"vote": 1, "weight": 0.3, "contribution": 30.0},
                        "xgboost": {"vote": 0, "weight": 0.2, "contribution": 0.0},
                    },
                ],
                "feature_importance": [{"spc": 0.5, "xgboost": 0.5}, {"spc": 0.6, "xgboost": 0.4}],
                "weighted_impact": [{"spc": 15.0, "xgboost": 10.0}, {"spc": 18.0, "xgboost": 0.0}],
            }
        )

        display_voting_metadata(df, signal_type="LONG")

        # (1) Assert that "LONG" (or signal_type) appears in the logged messages
        all_logged_messages = []
        for call in mock_log_data.call_args_list:
            if call.args:
                all_logged_messages.append(str(call.args[0]))
        for call in mock_log_progress.call_args_list:
            if call.args:
                all_logged_messages.append(str(call.args[0]))

        logged_text = " ".join(all_logged_messages)
        assert "LONG" in logged_text, f"'LONG' not found in logged messages. Messages: {logged_text}"

        # (2) Assert that each row's weighted_score and voting_breakdown entries appear
        # Check for weighted_score values (0.75 and 0.65 as percentages: 75.00% and 65.00%)
        assert "75.00%" in logged_text or "0.75" in logged_text, (
            "weighted_score 0.75 (75.00%) not found in logged messages"
        )
        assert "65.00%" in logged_text or "0.65" in logged_text, (
            "weighted_score 0.65 (65.00%) not found in logged messages"
        )

        # Check for voting_breakdown entries - should contain 'spc' and contribution values like "30.0"
        assert "spc" in logged_text.lower() or "SPC" in logged_text, (
            "'spc' not found in voting_breakdown logged messages"
        )
        assert "30.0" in logged_text or "30.00%" in logged_text, (
            "voting_breakdown contribution '30.0' not found in logged messages"
        )

        # (3) Assert that mock_log_data was called exactly for each signal (len(df) == 2)
        # Count calls that contain "Symbol:" to verify we have data for 2 signals
        symbol_calls = [
            call
            for call in mock_log_data.call_args_list
            if call.args and len(call.args) > 0 and "Symbol:" in str(call.args[0])
        ]
        assert len(symbol_calls) == 2, (
            f"Expected exactly 2 'Symbol:' log calls (one per signal), but found {len(symbol_calls)}"
        )

    @patch("cli.display.log_progress")
    @patch("cli.display.log_data")
    def test_display_voting_metadata_short_signals(self, mock_log_data, mock_log_progress):
        """Test display_voting_metadata for SHORT signals."""
        df = pd.DataFrame(
            {
                "symbol": ["BTC/USDT"],
                "weighted_score": [0.80],
                "voting_breakdown": [
                    {
                        "spc": {"vote": 0, "weight": 0.3, "contribution": 0.0},
                        "xgboost": {"vote": 0, "weight": 0.2, "contribution": 0.0},
                    }
                ],
                "feature_importance": [{"spc": 0.5, "xgboost": 0.5}],
                "weighted_impact": [{"spc": 0.0, "xgboost": 0.0}],
            }
        )

        display_voting_metadata(df, signal_type="SHORT")

        # Should log data for signals
        assert mock_log_data.called

        # Verify that the output contains "SHORT" to confirm signal_type is represented
        all_call_strings = []

        # Collect all call arguments from mock_log_data
        for call in mock_log_data.call_args_list:
            if call.args:
                all_call_strings.extend([str(arg) for arg in call.args])
            if call.kwargs:
                all_call_strings.extend([str(val) for val in call.kwargs.values()])

        # Collect all call arguments from mock_log_progress
        for call in mock_log_progress.call_args_list:
            if call.args:
                all_call_strings.extend([str(arg) for arg in call.args])
            if call.kwargs:
                all_call_strings.extend([str(val) for val in call.kwargs.values()])

        # Check if "SHORT" appears in any of the logged output
        assert any("SHORT" in call_str for call_str in all_call_strings), (
            "Expected 'SHORT' to appear in the logged output when signal_type='SHORT', but it was not found"
        )

    @patch("cli.display.log_progress")
    @patch("cli.display.log_data")
    def test_display_voting_metadata_with_spc_debug(self, mock_log_data, mock_log_progress):
        """Test display_voting_metadata with SPC debug enabled."""
        df = pd.DataFrame(
            {
                "symbol": ["BTC/USDT"],
                "weighted_score": [0.50],
                "voting_breakdown": [{"spc": {"vote": 0, "weight": 0.3, "contribution": 0.0}}],
                "feature_importance": [{"spc": 0.5}],
                "weighted_impact": [{"spc": 0.0}],
                "spc_cluster_transition_signal": [0],
                "spc_regime_following_signal": [0],
                "spc_mean_reversion_signal": [0],
                "spc_cluster_transition_strength": [0.0],
                "spc_regime_following_strength": [0.0],
                "spc_mean_reversion_strength": [0.0],
            }
        )

        display_voting_metadata(df, signal_type="LONG", show_spc_debug=True)

        # Should log debug information for SPC
        assert mock_log_data.called

        # Collect all logged messages as strings
        logged_messages = []
        for call in mock_log_data.call_args_list:
            if call.args:
                logged_messages.append(str(call.args[0]))
            if call.kwargs:
                logged_messages.extend([str(v) for v in call.kwargs.values() if isinstance(v, str)])

        # Verify SPC voting breakdown details are logged
        all_logs = " ".join(logged_messages)

        # Verify SPC indicator appears in logs
        assert "spc" in all_logs.lower() or "SPC" in all_logs, "SPC indicator not found in logged messages"

        # Verify SPC vote, weight, and contribution values are logged
        assert "0.3" in all_logs or "30.0%" in all_logs or "30%" in all_logs, (
            "SPC weight (0.3) not found in logged messages"
        )
        assert "0.0" in all_logs or "0.00%" in all_logs or "0%" in all_logs, (
            "SPC contribution (0.0) not found in logged messages"
        )
        assert "0.5" in all_logs or "50.0%" in all_logs or "50%" in all_logs, (
            "SPC feature importance (0.5) not found in logged messages"
        )

        # Verify SPC debug details are explicitly logged
        assert "[DEBUG]" in all_logs or "DEBUG" in all_logs, "SPC debug marker '[DEBUG]' not found in logged messages"
        assert "SPC Strategy Signals" in all_logs, (
            "SPC debug message 'SPC Strategy Signals' not found in logged messages"
        )

        # Verify SPC strategy signals (CT, RF, MR) are logged
        assert "CT=" in all_logs, "SPC Cluster Transition signal (CT=) not found in debug logs"
        assert "RF=" in all_logs, "SPC Regime Following signal (RF=) not found in debug logs"
        assert "MR=" in all_logs, "SPC Mean Reversion signal (MR=) not found in debug logs"

        # Verify strength values are logged (they should appear as strength=0.00)
        assert "strength=" in all_logs.lower(), "SPC strength values not found in debug logs"

        # Verify specific SPC signal values are logged
        assert "CT=0" in all_logs, "SPC Cluster Transition signal value (0) not found in debug logs"
        assert "RF=0" in all_logs, "SPC Regime Following signal value (0) not found in debug logs"
        assert "MR=0" in all_logs, "SPC Mean Reversion signal value (0) not found in debug logs"

    @patch("cli.display.log_progress")
    @patch("cli.display.log_data")
    def test_display_voting_metadata_limit_to_10(self, mock_log_data, mock_log_progress):
        """Test that display_voting_metadata only shows first 10 signals."""
        # Create DataFrame with more than 10 rows
        data = []
        for i in range(15):
            data.append(
                {
                    "symbol": f"SYMBOL{i}/USDT",
                    "weighted_score": 0.5 + i * 0.01,
                    "voting_breakdown": {"spc": {"vote": 1, "weight": 0.3, "contribution": 30.0}},
                    "feature_importance": {"spc": 0.5},
                    "weighted_impact": {"spc": 15.0},
                }
            )

        df = pd.DataFrame(data)

        display_voting_metadata(df, signal_type="LONG")

        # Should only process first 10 rows
        # Count number of "Symbol:" log calls to verify limit
        symbol_log_calls = [
            call
            for call in mock_log_data.call_args_list
            if call.args and len(call.args) > 0 and "Symbol:" in str(call.args[0])
        ]
        assert len(symbol_log_calls) == 10, (
            f"Expected exactly 10 signals to be processed, but found {len(symbol_log_calls)}"
        )

    @patch("cli.display.log_progress")
    @patch("cli.display.log_data")
    def test_display_voting_metadata_missing_fields(self, mock_log_data, mock_log_progress):
        """Test display_voting_metadata with missing optional fields."""
        df = pd.DataFrame(
            {
                "symbol": ["BTC/USDT"],
                "weighted_score": [0.75],
                "voting_breakdown": [{"spc": {"vote": 1, "weight": 0.3, "contribution": 30.0}}],
            }
        )

        # Should not raise error even with missing fields
        display_voting_metadata(df, signal_type="LONG")

        assert mock_log_data.called

        # Verify that the available fields are actually logged
        # Get all logged messages as strings
        all_logged_messages = []
        for call in mock_log_data.call_args_list:
            if call.args:
                all_logged_messages.extend([str(arg) for arg in call.args])
            if call.kwargs:
                all_logged_messages.extend([str(val) for val in call.kwargs.values()])

        all_logs = " ".join(all_logged_messages)

        # Verify symbol 'BTC/USDT' is logged
        assert "BTC/USDT" in all_logs, (
            f"Expected symbol 'BTC/USDT' to be logged, but it was not found. Logged messages: {all_logs}"
        )

        # Verify weighted_score 0.75 is logged (as 75.00% or 75%)
        assert "75.00%" in all_logs or "75%" in all_logs or "0.75" in all_logs, (
            f"Expected weighted_score 0.75 (75%) to be logged, but it was not found. Logged messages: {all_logs}"
        )

        # Verify voting_breakdown entry is logged (at least the 'spc' key should be present)
        assert "spc" in all_logs.lower() or "SPC" in all_logs, (
            f"Expected voting_breakdown entry 'spc' to be logged, but it was not found. Logged messages: {all_logs}"
        )

        # Verify voting_breakdown structure is present (vote, weight, contribution)
        assert "Voting Breakdown" in all_logs, (
            f"Expected 'Voting Breakdown' header to be logged, but it was not found. Logged messages: {all_logs}"
        )
