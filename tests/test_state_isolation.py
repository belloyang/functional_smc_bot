import unittest
import os
import json
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Mock config and other dependencies if needed
import app.bot as bot

class TestStateIsolation(unittest.TestCase):
    def setUp(self):
        # Clean up any existing test state files
        for f in os.listdir('.'):
            if f.startswith('trade_state_') and f.endswith('.json'):
                os.remove(f)
        if os.path.exists('trade_state.json'):
            os.remove('trade_state.json')

    def tearDown(self):
        # Cleanup after tests
        for f in os.listdir('.'):
            if f.startswith('trade_state_') and f.endswith('.json'):
                os.remove(f)
        if os.path.exists('trade_state.json'):
            os.remove('trade_state.json')

    def test_default_state_path(self):
        # Default should be trade_state.json
        path = bot.get_state_file_path()
        self.assertEqual(path, "trade_state.json")

    def test_symbol_specific_path(self):
        # Symbol-specific should be trade_state_SYMBOL.json
        path = bot.get_state_file_path("AAPL")
        self.assertEqual(path, "trade_state_AAPL.json")

    def test_save_load_isolation(self):
        # Save to AAPL
        state_aapl = {"AAPL": {"virtual_stop": 10.0}}
        bot.save_trade_state(state_aapl, "AAPL")
        
        # Save to MSFT
        state_msft = {"MSFT": {"virtual_stop": 20.0}}
        bot.save_trade_state(state_msft, "MSFT")
        
        # Load AAPL
        loaded_aapl = bot.load_trade_state("AAPL")
        self.assertEqual(loaded_aapl["AAPL"]["virtual_stop"], 10.0)
        self.assertNotIn("MSFT", loaded_aapl)
        
        # Load MSFT
        loaded_msft = bot.load_trade_state("MSFT")
        self.assertEqual(loaded_msft["MSFT"]["virtual_stop"], 20.0)
        self.assertNotIn("AAPL", loaded_msft)
        
        # Verify files exist
        self.assertTrue(os.path.exists("trade_state_AAPL.json"))
        self.assertTrue(os.path.exists("trade_state_MSFT.json"))
        self.assertFalse(os.path.exists("trade_state.json"))

    def test_override_path(self):
        # Set a global args mock
        class ArgsMock:
            state_file = "custom_state.json"
        
        bot.args = ArgsMock()
        
        path = bot.get_state_file_path("AAPL")
        self.assertEqual(path, "custom_state.json")
        
        # Cleanup the global mock
        del bot.args

if __name__ == '__main__':
    unittest.main()
