#!/usr/bin/env python
"""
AI Trader - Main Entry Point

Usage:
    python main.py analyze AAPL --strategy technical
    python main.py backtest AAPL --period 1y
    python main.py portfolio show
    python main.py schedule AAPL MSFT --interval 15
    python main.py dashboard
    
For more options:
    python main.py --help
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trader.cli import main

if __name__ == '__main__':
    sys.exit(main())
