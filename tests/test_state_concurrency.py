import sys
import os
from pathlib import Path

# Add project root to sys.path
root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

import json
import time
import unittest
from unittest.mock import MagicMock, patch

# Mock dependencies before importing bot
sys.modules['alpaca'] = MagicMock()
sys.modules['alpaca.trading'] = MagicMock()
sys.modules['alpaca.trading.client'] = MagicMock()
sys.modules['alpaca.trading.requests'] = MagicMock()
sys.modules['alpaca.trading.enums'] = MagicMock()
sys.modules['alpaca.data'] = MagicMock()
sys.modules['alpaca.data.historical'] = MagicMock()
sys.modules['alpaca.data.requests'] = MagicMock()
sys.modules['alpaca.data.timeframe'] = MagicMock()
sys.modules['pandas_ta'] = MagicMock()

# Mock argparse.Namespace to simulate CLI arguments
class MockArgs:
    def __init__(self, state_file=None):
        self.state_file = state_file

# Import the functions directly from bot.py
import app.bot as bot

class TestStateConcurrency(unittest.TestCase):
    def setUp(self):
        self.test_symbol = "TEST"
        self.state_file = f"trade_state_{self.test_symbol}.json"
        if os.path.exists(self.state_file):
            os.remove(self.state_file)
        # Mock bot.args
        bot.args = MockArgs()

    def tearDown(self):
        if os.path.exists(self.state_file):
            os.remove(self.state_file)
        if os.path.exists("custom_state.json"):
            os.remove("custom_state.json")

    def test_state_file_path_default(self):
        path = bot.get_state_file_path(self.test_symbol)
        self.assertEqual(path, f"trade_state_{self.test_symbol}.json")

    def test_state_file_path_override(self):
        bot.args.state_file = "custom_state.json"
        path = bot.get_state_file_path(self.test_symbol)
        self.assertEqual(path, "custom_state.json")

    def test_save_and_load_state(self):
        state = {"TEST": {"daily_trade_count": 1}}
        bot.save_trade_state(state, self.test_symbol)
        
        loaded_state = bot.load_trade_state(self.test_symbol)
        self.assertEqual(loaded_state, state)

    @patch("builtins.open")
    def test_load_retry_on_lock(self, mock_open):
        # Simulate PermissionError on first 2 calls, success on 3rd
        state_content = json.dumps({"TEST": {"count": 1}})
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = state_content
        
        # side_effect: raise IOError, raise IOError, return mock_file
        # Note: open() returns the mock_file which is used as context manager
        mock_open.side_effect = [IOError("Permission denied"), IOError("Permission denied"), mock_file]
        
        # We need to make sure os.path.exists returns True
        with patch("os.path.exists", return_value=True):
            with patch("time.sleep") as mock_sleep:
                state = bot.load_trade_state(self.test_symbol)
                self.assertEqual(mock_open.call_count, 3)
                self.assertEqual(mock_sleep.call_count, 2)
                # json.load calls f.read()
                # Actually json.load(f) calls f.read() or iterates
                # If we use json.load(f), it calls f.read()
        
    @patch("builtins.open")
    def test_save_retry_on_lock(self, mock_open):
        state = {"TEST": {"count": 1}}
        mock_open.side_effect = [IOError("Permission denied"), MagicMock()]
        
        with patch("time.sleep") as mock_sleep:
            bot.save_trade_state(state, self.test_symbol)
            self.assertEqual(mock_open.call_count, 2)
            self.assertEqual(mock_sleep.call_count, 1)

if __name__ == "__main__":
    unittest.main()
