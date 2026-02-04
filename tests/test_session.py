"""Tests for TradingSession class."""

import time
import sys
from datetime import datetime


class TradingSession:
    """Manages trading session state and statistics."""
    
    def __init__(self, duration_hours=None, max_trades=None):
        self.start_time = datetime.now()
        self.duration_hours = duration_hours
        self.max_trades = max_trades
        self.trades_executed = 0
        self.should_stop = False
        
    def record_trade(self):
        """Record that a trade was executed."""
        self.trades_executed += 1
        
    def should_continue(self):
        """Check if the session should continue running."""
        if self.should_stop:
            return False
            
        # Check duration limit
        if self.duration_hours is not None:
            elapsed_hours = (datetime.now() - self.start_time).total_seconds() / 3600
            if elapsed_hours >= self.duration_hours:
                return False
                
        # Check trades limit
        if self.max_trades is not None:
            if self.trades_executed >= self.max_trades:
                return False
                
        return True
    
    def get_summary(self):
        """Get session summary statistics."""
        elapsed = datetime.now() - self.start_time
        hours = int(elapsed.total_seconds() // 3600)
        minutes = int((elapsed.total_seconds() % 3600) // 60)
        
        summary = [
            "=" * 50,
            "SESSION SUMMARY",
            "=" * 50,
            f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Duration: {hours}h {minutes}m",
            f"Trades Executed: {self.trades_executed}",
        ]
        
        if self.max_trades:
            summary.append(f"Trade Limit: {self.max_trades}")
        if self.duration_hours:
            summary.append(f"Duration Limit: {self.duration_hours} hours")
            
        summary.append("=" * 50)
        return "\n".join(summary)
    
    def request_stop(self):
        """Request session to stop gracefully."""
        self.should_stop = True


def test_session_duration_limit():
    """Test that session stops after duration limit."""
    print("Testing session duration limit...")
    
    # Create session with very short duration (0.001 hours ~ 3.6 seconds)
    session = TradingSession(duration_hours=0.001)
    
    assert session.should_continue()
    print("✓ Session initially continues")
    
    # Wait for duration to expire
    time.sleep(4)
    
    assert not session.should_continue()
    print("✓ Session stops after duration limit")
    print()


def test_session_trade_limit():
    """Test that session stops after trade limit."""
    print("Testing session trade limit...")
    
    session = TradingSession(max_trades=3)
    
    assert session.should_continue()
    assert session.trades_executed == 0
    print("✓ Session starts with 0 trades")
    
    # Record 3 trades
    session.record_trade()
    assert session.trades_executed == 1
    assert session.should_continue()
    
    session.record_trade()
    assert session.trades_executed == 2
    assert session.should_continue()
    
    session.record_trade()
    assert session.trades_executed == 3
    assert not session.should_continue()
    print("✓ Session stops after max trades reached")
    print()


def test_session_manual_stop():
    """Test that session stops when manually requested."""
    print("Testing manual stop...")
    
    session = TradingSession()
    
    assert session.should_continue()
    print("✓ Session initially continues")
    
    session.request_stop()
    assert not session.should_continue()
    print("✓ Session stops after manual request")
    print()


def test_session_summary():
    """Test that session generates summary correctly."""
    print("Testing session summary...")
    
    session = TradingSession(duration_hours=2, max_trades=5)
    session.record_trade()
    session.record_trade()
    
    summary = session.get_summary()
    
    assert "SESSION SUMMARY" in summary
    assert "Trades Executed: 2" in summary
    assert "Trade Limit: 5" in summary
    assert "Duration Limit: 2 hours" in summary
    print("✓ Session summary contains expected information")
    print()


def test_session_unlimited():
    """Test that session with no limits continues."""
    print("Testing unlimited session...")
    
    session = TradingSession()
    
    # Unlimited session should always continue (until manually stopped)
    for i in range(10):
        session.record_trade()
        assert session.should_continue()
    
    print("✓ Unlimited session continues indefinitely")
    print()


if __name__ == "__main__":
    print("Running TradingSession tests...\n")
    print("=" * 50)
    
    try:
        test_session_duration_limit()
        test_session_trade_limit()
        test_session_manual_stop()
        test_session_summary()
        test_session_unlimited()
        
        print("=" * 50)
        print("✅ All tests passed!")
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        sys.exit(1)
