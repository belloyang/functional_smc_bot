#!/usr/bin/env python3.12
"""
Demo script to test the session management functionality.
This simulates the bot running with different session configurations.
"""

import time
from datetime import datetime
import signal
import sys


class MockTradingSession:
    """Mock version of TradingSession for demo purposes."""
    
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
        seconds = int(elapsed.total_seconds() % 60)
        
        summary = [
            "=" * 50,
            "SESSION SUMMARY",
            "=" * 50,
            f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Duration: {hours}h {minutes}m {seconds}s",
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


def demo_time_limited():
    """Demo: Session with time limit."""
    print("\n" + "=" * 50)
    print("DEMO 1: Time-Limited Session (10 seconds)")
    print("=" * 50)
    
    session = MockTradingSession(duration_hours=10/3600)  # 10 seconds
    
    # Setup signal handler
    def signal_handler(signum, frame):
        print("\n🛑 Shutdown signal received. Stopping gracefully...")
        session.request_stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    iteration = 0
    while session.should_continue():
        iteration += 1
        print(f"[{iteration}] Running... (Press Ctrl+C to stop early)")
        
        # Simulate random trade
        if iteration % 3 == 0:
            session.record_trade()
            print(f"    💰 Simulated trade #{session.trades_executed}")
        
        time.sleep(2)
    
    print("\n" + session.get_summary())


def demo_trade_limited():
    """Demo: Session with trade limit."""
    print("\n" + "=" * 50)
    print("DEMO 2: Trade-Limited Session (max 5 trades)")
    print("=" * 50)
    
    session = MockTradingSession(max_trades=5)
    
    iteration = 0
    while session.should_continue():
        iteration += 1
        print(f"[{iteration}] Scanning for signals...")
        
        # Simulate finding a signal every few iterations
        if iteration % 2 == 0:
            session.record_trade()
            print(f"    🚀 Signal detected! Trade #{session.trades_executed} executed.")
        
        time.sleep(1)
    
    print("\n" + session.get_summary())


def demo_combined():
    """Demo: Session with both limits."""
    print("\n" + "=" * 50)
    print("DEMO 3: Combined Limits (10s OR 3 trades)")
    print("=" * 50)
    
    session = MockTradingSession(duration_hours=10/3600, max_trades=3)
    
    iteration = 0
    while session.should_continue():
        iteration += 1
        print(f"[{iteration}] Running... (Trades: {session.trades_executed}/3)")
        
        # Simulate occasional trade
        if iteration % 4 == 0:
            session.record_trade()
            print(f"    💰 Trade #{session.trades_executed} executed.")
        
        time.sleep(1.5)
    
    print("\n" + session.get_summary())


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Session Management Demo")
    print("=" * 50)
    
    try:
        demo_time_limited()
        time.sleep(1)
        
        demo_trade_limited()
        time.sleep(1)
        
        demo_combined()
        
        print("\n✅ All demos completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user.")
        sys.exit(0)
