# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.1-ibkr] - 2026-02-08

### Changed
- **Reduced maximum option allocation percentage from 20% to 15%**: The `OPTIONS_ALLOCATION_PCT` configuration parameter has been lowered from 0.20 (20%) to 0.15 (15%) to improve risk management for options trading. This change affects the global maximum percentage of equity that can be allocated to all option premiums.

### Impact
- Users running options strategies will now have a more conservative allocation limit
- This change helps reduce overall portfolio risk exposure from options positions
- Default behavior for `--option-budget` parameter remains at the new 15% limit unless explicitly overridden

## [1.3.0-ibkr] - 2026-02-08

### Added
- Initial release with comprehensive trading bot features
- Backtesting support with separate stock/option allocation
- Option trading with dynamic budgeting
- Stock trading with risk-based sizing
- Multi-instance safety for running multiple tickers concurrently
- Risk management (Daily caps, Per-ticker limits)
- Order management (Bracket orders with TP/SL)
- Session management with duration and trade limits
- Graceful shutdown with session summaries
