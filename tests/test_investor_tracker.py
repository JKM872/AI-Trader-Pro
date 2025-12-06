"""
Tests for Investor Portfolio Tracker module.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
import json

from trader.data.investor_tracker import (
    PortfolioTracker,
    InvestorProfile,
    FAMOUS_INVESTORS,
)


# ============== Fixtures ==============

@pytest.fixture
def mock_httpx_response():
    """Create mock httpx response factory."""
    def _create(data, status_code=200):
        mock = Mock()
        mock.status_code = status_code
        mock.json.return_value = data
        mock.text = json.dumps(data) if isinstance(data, dict) else str(data)
        mock.raise_for_status = Mock()
        if status_code >= 400:
            mock.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
        return mock
    return _create


# ============== FAMOUS_INVESTORS Tests ==============

class TestFamousInvestors:
    """Tests for FAMOUS_INVESTORS constant."""
    
    def test_famous_investors_count(self):
        """Test we have 15+ famous investors defined."""
        assert len(FAMOUS_INVESTORS) >= 15
    
    def test_famous_investors_structure(self):
        """Test each investor has required fields."""
        for key, investor in FAMOUS_INVESTORS.items():
            assert isinstance(investor, InvestorProfile)
            assert investor.name is not None
            assert investor.cik is not None
            assert investor.fund_name is not None
    
    def test_famous_investors_includes_buffett(self):
        """Test Buffett is in the list."""
        assert "warren_buffett" in FAMOUS_INVESTORS
        assert FAMOUS_INVESTORS["warren_buffett"].name == "Warren Buffett"
    
    def test_famous_investors_includes_dalio(self):
        """Test Dalio is in the list."""
        assert "ray_dalio" in FAMOUS_INVESTORS
        assert FAMOUS_INVESTORS["ray_dalio"].name == "Ray Dalio"
    
    def test_famous_investors_cik_format(self):
        """Test CIK numbers are properly formatted."""
        for key, investor in FAMOUS_INVESTORS.items():
            assert investor.cik.startswith("0")
            assert len(investor.cik) == 10


# ============== InvestorProfile Tests ==============

class TestInvestorProfile:
    """Tests for InvestorProfile dataclass."""
    
    def test_investor_profile_creation(self):
        """Test creating InvestorProfile."""
        profile = InvestorProfile(
            name="Test Investor",
            cik="0001234567",
            fund_name="Test Fund",
            strategy="Value Investing",
            description="Test description"
        )
        
        assert profile.name == "Test Investor"
        assert profile.cik == "0001234567"
        assert profile.fund_name == "Test Fund"


# ============== PortfolioTracker Tests ==============

class TestPortfolioTracker:
    """Tests for PortfolioTracker class."""
    
    def test_tracker_initialization(self):
        """Test tracker initializes correctly."""
        tracker = PortfolioTracker()
        assert tracker is not None
        tracker.client.close()
    
    def test_get_all_investors(self):
        """Test getting all tracked investors."""
        tracker = PortfolioTracker()
        
        # Should be able to access all investors
        assert len(FAMOUS_INVESTORS) >= 15
        tracker.client.close()
    
    @patch('httpx.Client')
    def test_get_investor_holdings_success(self, mock_client_class, mock_httpx_response):
        """Test fetching investor holdings."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_client.get.return_value = mock_httpx_response({
            "cik": "0001067983",
            "name": "BERKSHIRE HATHAWAY INC",
            "filings": {
                "recent": {
                    "form": ["13F-HR", "10-K", "8-K"],
                    "filingDate": ["2024-01-15", "2024-01-10", "2024-01-05"],
                    "accessionNumber": ["0001234567-24-000001", "0001234567-24-000002", "0001234567-24-000003"]
                }
            }
        })
        
        tracker = PortfolioTracker()
        tracker.client = mock_client
        
        result = tracker.get_investor_holdings("warren_buffett")
        
        assert result is not None
        assert result.get("investor") == "Warren Buffett"
    
    def test_get_investor_holdings_unknown(self):
        """Test getting holdings for unknown investor."""
        tracker = PortfolioTracker()
        
        with pytest.raises(ValueError, match="Unknown investor"):
            tracker.get_investor_holdings("unknown_person")
        
        tracker.client.close()
    
    def test_tracker_has_required_methods(self):
        """Test tracker has required methods."""
        tracker = PortfolioTracker()
        
        assert hasattr(tracker, 'get_investor_holdings')
        assert hasattr(tracker, '_parse_13f_holdings')
        
        tracker.client.close()


# ============== SEC Integration Tests ==============

class TestSECIntegration:
    """Tests for SEC Edgar integration."""
    
    def test_sec_base_url(self):
        """Test SEC Edgar base URL is correct."""
        tracker = PortfolioTracker()
        assert tracker.SEC_EDGAR_BASE == "https://data.sec.gov"
        tracker.client.close()
    
    def test_investor_ciks_are_valid(self):
        """Test all investor CIKs are valid format."""
        for key, investor in FAMOUS_INVESTORS.items():
            # CIK should be 10 digits with leading zeros
            assert len(investor.cik) == 10
            assert investor.cik.isdigit()


# ============== Integration Tests ==============

class TestInvestorTrackerIntegration:
    """Integration tests for investor tracker."""
    
    def test_tracker_works_without_network(self):
        """Test tracker initializes without network."""
        tracker = PortfolioTracker()
        assert tracker is not None
        tracker.client.close()
    
    def test_all_famous_investors_valid(self):
        """Test all famous investors have valid data."""
        for key, investor in FAMOUS_INVESTORS.items():
            assert isinstance(investor.name, str)
            assert len(investor.name) > 0
            assert isinstance(investor.strategy, str)
            assert len(investor.strategy) > 0
