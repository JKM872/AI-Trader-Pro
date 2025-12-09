"""
Integration tests for AI Trader Pro.

Tests end-to-end workflows across multiple components.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta

from trader.data.fetcher import DataFetcher
from trader.strategies.technical import TechnicalStrategy
from trader.strategies.momentum import MomentumStrategy
from trader.strategies.scanner import SignalScanner
from trader.analysis.opportunity_scorer import OpportunityScorer
from trader.portfolio.paper_trader import PaperAccount, OrderSide


class TestFullAnalysisFlow:
    """Test complete analysis workflow."""
    
    @pytest.mark.slow
    def test_fetch_analyze_score_flow(self):
        """Test: Fetch data → Analyze → Score opportunity."""
        symbol = "AAPL"
        
        # Step 1: Fetch data
        fetcher = DataFetcher()
        df = fetcher.get_stock_data(symbol, period='6mo')
        
        assert not df.empty
        assert 'Close' in df.columns
        
        # Step 2: Analyze with strategy
        strategy = TechnicalStrategy()
        signal = strategy.generate_signal(symbol, df)
        
        assert signal is not None
        assert signal.symbol == symbol
        assert signal.confidence >= 0
        
        # Step 3: Score opportunity
        scorer = OpportunityScorer()
        score = scorer.score_stock(symbol)
        
        assert score.symbol == symbol
        assert 0 <= score.total_score <= 100
        assert score.recommendation in ["Strong Buy", "Buy", "Hold", "Avoid", "High Risk"]


class TestMultiStrategyAnalysis:
    """Test running multiple strategies on same symbol."""
    
    @pytest.mark.slow
    def test_all_strategies_on_symbol(self):
        """Test all strategies work on a single symbol."""
        symbol = "MSFT"
        fetcher = DataFetcher()
        df = fetcher.get_stock_data(symbol, period='6mo')
        
        strategies = [
            TechnicalStrategy(),
            MomentumStrategy()
        ]
        
        signals = []
        for strategy in strategies:
            signal = strategy.generate_signal(symbol, df)
            if signal:
                signals.append(signal)
        
        # At least some strategies should produce signals
        assert len(signals) > 0
        
        # All signals should be for the same symbol
        for sig in signals:
            assert sig.symbol == symbol


class TestScannerIntegration:
    """Test scanner with multiple symbols."""
    
    @pytest.mark.slow
    def test_scanner_watchlist(self):
        """Test scanning a watchlist."""
        symbols = ["AAPL", "MSFT"]
        scanner = SignalScanner()
        
        results = scanner.scan_watchlist(symbols)
        
        # Should return dict
        assert isinstance(results, dict)
        
        # Each symbol should have results (could be empty list)
        for sym in symbols:
            assert sym in results


class TestPaperTradingIntegration:
    """Test paper trading with signals."""
    
    def setup_method(self):
        """Setup for each test."""
        self.account = PaperAccount(initial_balance=10000.0)
    
    @pytest.mark.slow
    def test_signal_to_trade_flow(self):
        """Test: Generate signal → Execute paper trade → Track position."""
        symbol = "AAPL"
        
        # Generate signal
        fetcher = DataFetcher()
        df = fetcher.get_stock_data(symbol, period='1mo')
        
        strategy = TechnicalStrategy()
        signal = strategy.generate_signal(symbol, df)
        
        # If signal is BUY, execute paper trade
        if signal and signal.signal_type.value == "BUY":
            current_price = df['Close'].iloc[-1]
            quantity = 10
            
            success, msg = self.account.execute_trade(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=quantity,
                price=current_price
            )
            
            assert success is True
            assert symbol in self.account.positions
            
            # Update position price
            self.account.update_prices({symbol: current_price * 1.05})  # 5% gain
            
            # Check P&L
            pos = self.account.positions[symbol]
            assert pos.unrealized_pnl > 0


class TestErrorHandling:
    """Test system resilience to errors."""
    
    def test_invalid_symbol(self):
        """Test handling of invalid symbol."""
        fetcher = DataFetcher()
        df = fetcher.get_stock_data("INVALID_SYMBOL_XYZ", period='1mo')
        
        # Should return empty DataFrame, not crash
        assert isinstance(df, pd.DataFrame)
    
    def test_insufficient_data(self):
        """Test strategy with insufficient data."""
        strategy = TechnicalStrategy()
        
        # Create minimal DataFrame
        df = pd.DataFrame({
            'Open': [100],
            'High': [101],
            'Low': [99],
            'Close': [100.5],
            'Volume': [1000000]
        })
        
        signal = strategy.generate_signal("TEST", df)
        
        # Should handle gracefully (return HOLD or None)
        assert signal is None or signal.signal_type.value == "HOLD"
    
    @pytest.mark.slow
    def test_api_failure_resilience(self):
        """Test system handles API failures gracefully."""
        # Try to fetch data for multiple symbols, some may fail
        symbols = ["AAPL", "INVALID", "MSFT", "BADDATA123"]
        
        fetcher = DataFetcher()
        successful = 0
        
        for sym in symbols:
            try:
                df = fetcher.get_stock_data(sym, period='1mo')
                if not df.empty:
                    successful += 1
            except Exception:
                pass  # Should not crash
        
        # At least valid symbols should work
        assert successful >= 2


class TestDataConsistency:
    """Test data consistency across components."""
    
    @pytest.mark.slow
    def test_price_consistency(self):
        """Test that current price is consistent across components."""
        symbol = "AAPL"
        
        # Get price from data fetcher
        fetcher = DataFetcher()
        df = fetcher.get_stock_data(symbol, period='1d')
        fetcher_price = df['Close'].iloc[-1] if not df.empty else None
        
        # Get current price
        current_price = fetcher.get_current_price(symbol)
        
        # Prices should be reasonably close (within 5%)
        if fetcher_price and current_price:
            diff_pct = abs(fetcher_price - current_price) / fetcher_price * 100
            assert diff_pct < 5.0


class TestPerformanceMetrics:
    """Test that components complete in reasonable time."""
    
    @pytest.mark.slow
    def test_single_analysis_performance(self):
        """Test single stock analysis completes in reasonable time."""
        import time
        
        start = time.time()
        
        fetcher = DataFetcher()
        df = fetcher.get_stock_data("AAPL", period='6mo')
        
        strategy = TechnicalStrategy()
        signal = strategy.generate_signal("AAPL", df)
        
        elapsed = time.time() - start
        
        # Should complete in less than 10 seconds
        assert elapsed < 10.0
    
    @pytest.mark.slow
    def test_opportunity_scoring_performance(self):
        """Test opportunity scoring completes in reasonable time."""
        import time
        
        start = time.time()
        
        scorer = OpportunityScorer()
        score = scorer.score_stock("AAPL")
        
        elapsed = time.time() - start
        
        # Should complete in less than 15 seconds
        assert elapsed < 15.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
