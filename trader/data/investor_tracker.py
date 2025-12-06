"""
Investor Portfolio Tracker - Track famous investors and their portfolios.

Sources:
- SEC 13F filings (institutional holdings)
- Form 4 (insider trades)
- Whale tracking services
- Famous investor portfolios (Buffett, Dalio, etc.)
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import httpx

logger = logging.getLogger(__name__)


@dataclass
class InvestorProfile:
    """Profile of a famous investor."""
    name: str
    cik: str  # SEC Central Index Key
    fund_name: str
    strategy: str
    description: str
    holdings_value: Optional[float] = None
    last_updated: Optional[str] = None


# Famous investors with their SEC CIK numbers
FAMOUS_INVESTORS: Dict[str, InvestorProfile] = {
    "warren_buffett": InvestorProfile(
        name="Warren Buffett",
        cik="0001067983",
        fund_name="Berkshire Hathaway",
        strategy="Value Investing",
        description="Oracle of Omaha, long-term value investor"
    ),
    "ray_dalio": InvestorProfile(
        name="Ray Dalio",
        cik="0001350694",
        fund_name="Bridgewater Associates",
        strategy="All Weather / Risk Parity",
        description="Largest hedge fund, macro strategies"
    ),
    "carl_icahn": InvestorProfile(
        name="Carl Icahn",
        cik="0000921669",
        fund_name="Icahn Enterprises",
        strategy="Activist Investing",
        description="Corporate activist investor"
    ),
    "david_tepper": InvestorProfile(
        name="David Tepper",
        cik="0001656456",
        fund_name="Appaloosa Management",
        strategy="Distressed Debt / Event-Driven",
        description="Distressed securities specialist"
    ),
    "bill_ackman": InvestorProfile(
        name="Bill Ackman",
        cik="0001336528",
        fund_name="Pershing Square",
        strategy="Activist Value Investing",
        description="Concentrated activist positions"
    ),
    "george_soros": InvestorProfile(
        name="George Soros",
        cik="0001029160",
        fund_name="Soros Fund Management",
        strategy="Global Macro",
        description="Legendary macro trader"
    ),
    "paul_singer": InvestorProfile(
        name="Paul Singer",
        cik="0001061165",
        fund_name="Elliott Management",
        strategy="Activist / Distressed",
        description="Activist hedge fund"
    ),
    "stan_druckenmiller": InvestorProfile(
        name="Stanley Druckenmiller",
        cik="0001536411",
        fund_name="Duquesne Family Office",
        strategy="Global Macro",
        description="Former Soros partner, macro investor"
    ),
    "chase_coleman": InvestorProfile(
        name="Chase Coleman",
        cik="0001167483",
        fund_name="Tiger Global",
        strategy="Growth / Tech",
        description="Tiger cub, technology focus"
    ),
    "michael_burry": InvestorProfile(
        name="Michael Burry",
        cik="0001649339",
        fund_name="Scion Asset Management",
        strategy="Value / Contrarian",
        description="The Big Short fame, contrarian"
    ),
    "cathie_wood": InvestorProfile(
        name="Cathie Wood",
        cik="0001803926",
        fund_name="ARK Invest",
        strategy="Disruptive Innovation",
        description="Innovation-focused ETF manager"
    ),
    "ken_griffin": InvestorProfile(
        name="Ken Griffin",
        cik="0001397545",
        fund_name="Citadel",
        strategy="Multi-Strategy",
        description="Largest hedge fund, multiple strategies"
    ),
    "dan_loeb": InvestorProfile(
        name="Daniel Loeb",
        cik="0001040273",
        fund_name="Third Point",
        strategy="Event-Driven / Activist",
        description="Activist and event-driven investor"
    ),
    "lee_ainslie": InvestorProfile(
        name="Lee Ainslie",
        cik="0001009207",
        fund_name="Maverick Capital",
        strategy="Long/Short Equity",
        description="Tiger cub, fundamental equity"
    ),
    "seth_klarman": InvestorProfile(
        name="Seth Klarman",
        cik="0001061768",
        fund_name="Baupost Group",
        strategy="Value Investing",
        description="Deep value, safety margin focus"
    ),
}


class PortfolioTracker:
    """
    Track institutional and famous investor portfolios using SEC filings.
    """
    
    SEC_EDGAR_BASE = "https://data.sec.gov"
    SEC_FULL_TEXT = "https://efts.sec.gov/LATEST/search-index"
    
    def __init__(self):
        self.client = httpx.Client(
            timeout=30.0,
            headers={
                "User-Agent": "AI-Trader/1.0 (Research Purpose)",
                "Accept": "application/json"
            }
        )
    
    def get_investor_holdings(self, investor_key: str) -> Dict[str, Any]:
        """
        Get current holdings for a famous investor from their 13F filing.
        
        Args:
            investor_key: Key from FAMOUS_INVESTORS dict
        
        Returns:
            Dictionary with holdings data
        """
        if investor_key not in FAMOUS_INVESTORS:
            raise ValueError(f"Unknown investor: {investor_key}")
        
        investor = FAMOUS_INVESTORS[investor_key]
        
        try:
            # Get recent filings
            response = self.client.get(
                f"{self.SEC_EDGAR_BASE}/submissions/CIK{investor.cik}.json"
            )
            response.raise_for_status()
            data = response.json()
            
            # Find most recent 13F-HR filing
            filings = data.get("filings", {}).get("recent", {})
            forms = filings.get("form", [])
            dates = filings.get("filingDate", [])
            accessions = filings.get("accessionNumber", [])
            
            latest_13f = None
            for i, form in enumerate(forms):
                if form == "13F-HR":
                    latest_13f = {
                        "date": dates[i],
                        "accession": accessions[i].replace("-", "")
                    }
                    break
            
            if not latest_13f:
                return {
                    "investor": investor.name,
                    "fund": investor.fund_name,
                    "error": "No 13F filing found"
                }
            
            # Get the 13F holdings
            holdings = self._parse_13f_holdings(investor.cik, latest_13f["accession"])
            
            return {
                "investor": investor.name,
                "fund": investor.fund_name,
                "strategy": investor.strategy,
                "filing_date": latest_13f["date"],
                "holdings": holdings,
                "total_positions": len(holdings),
                "total_value": sum(h.get("value", 0) for h in holdings)
            }
            
        except Exception as e:
            logger.error(f"Failed to get holdings for {investor_key}: {e}")
            return {
                "investor": investor.name,
                "error": str(e)
            }
    
    def _parse_13f_holdings(self, cik: str, accession: str) -> List[Dict]:
        """Parse holdings from 13F-HR XML/JSON."""
        holdings = []
        
        try:
            # Try to get the information table
            url = f"{self.SEC_EDGAR_BASE}/Archives/edgar/data/{cik.lstrip('0')}/{accession}"
            
            # List files in the filing
            response = self.client.get(f"{url}/index.json")
            if response.status_code == 200:
                index = response.json()
                
                # Find the information table file
                for item in index.get("directory", {}).get("item", []):
                    if "infotable" in item.get("name", "").lower() or \
                       "information" in item.get("name", "").lower():
                        # This would require XML parsing
                        # For now, return basic info
                        pass
            
            # Alternative: Use SEC full-text search
            # This is a simplified version - full implementation would parse XML
            
        except Exception as e:
            logger.warning(f"Failed to parse 13F: {e}")
        
        return holdings
    
    def get_all_famous_investor_holdings(self) -> Dict[str, Dict]:
        """Get holdings for all famous investors."""
        results = {}
        for key, investor in FAMOUS_INVESTORS.items():
            try:
                results[key] = self.get_investor_holdings(key)
            except Exception as e:
                results[key] = {"error": str(e), "investor": investor.name}
        return results
    
    def get_stock_institutional_owners(self, ticker: str) -> Dict[str, Any]:
        """
        Get all institutional owners of a stock.
        
        Args:
            ticker: Stock symbol
        
        Returns:
            Dictionary with institutional ownership data
        """
        import yfinance as yf
        
        try:
            stock = yf.Ticker(ticker)
            
            # Institutional holders
            inst_holders = stock.institutional_holders
            major_holders = stock.major_holders
            
            result = {
                "ticker": ticker,
                "institutional_holders": [],
                "major_holders": {}
            }
            
            if inst_holders is not None and not inst_holders.empty:
                result["institutional_holders"] = inst_holders.to_dict("records")
                result["total_institutional_shares"] = inst_holders["Shares"].sum() if "Shares" in inst_holders.columns else 0
            
            if major_holders is not None and not major_holders.empty:
                result["major_holders"] = {
                    "insiders_pct": major_holders.iloc[0, 0] if len(major_holders) > 0 else None,
                    "institutions_pct": major_holders.iloc[1, 0] if len(major_holders) > 1 else None,
                    "institutions_float_pct": major_holders.iloc[2, 0] if len(major_holders) > 2 else None,
                    "institutions_count": major_holders.iloc[3, 0] if len(major_holders) > 3 else None
                }
            
            # Check if any famous investor holds this stock
            result["famous_investors"] = []
            for key, investor in FAMOUS_INVESTORS.items():
                if inst_holders is not None and "Holder" in inst_holders.columns:
                    for _, row in inst_holders.iterrows():
                        holder_name = str(row.get("Holder", "")).upper()
                        if investor.fund_name.upper() in holder_name:
                            result["famous_investors"].append({
                                "investor": investor.name,
                                "fund": investor.fund_name,
                                "shares": row.get("Shares"),
                                "value": row.get("Value"),
                                "pct_out": row.get("% Out")
                            })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get institutional owners for {ticker}: {e}")
            return {"ticker": ticker, "error": str(e)}
    
    def get_insider_transactions(self, ticker: str, days: int = 90) -> List[Dict]:
        """
        Get insider transactions for a stock.
        
        Args:
            ticker: Stock symbol
            days: Number of days to look back
        
        Returns:
            List of insider transactions
        """
        import yfinance as yf
        
        try:
            stock = yf.Ticker(ticker)
            insider_trans = stock.insider_transactions
            insider_roster = stock.insider_roster_holders
            
            transactions = []
            
            if insider_trans is not None and not insider_trans.empty:
                for _, row in insider_trans.iterrows():
                    transactions.append({
                        "insider": row.get("Insider Trading"),
                        "relation": row.get("Relationship"),
                        "date": str(row.get("Start Date")),
                        "transaction": row.get("Transaction"),
                        "shares": row.get("Shares"),
                        "value": row.get("Value"),
                        "text": row.get("Text")
                    })
            
            return {
                "ticker": ticker,
                "transactions": transactions[:50],
                "total_transactions": len(transactions),
                "insider_roster": insider_roster.to_dict("records") if insider_roster is not None and not insider_roster.empty else []
            }
            
        except Exception as e:
            logger.error(f"Failed to get insider transactions for {ticker}: {e}")
            return {"ticker": ticker, "error": str(e)}
    
    def track_holdings_changes(self, investor_key: str) -> Dict[str, Any]:
        """
        Track changes in an investor's holdings between quarters.
        
        Compares last two 13F filings to show:
        - New positions
        - Closed positions  
        - Increased positions
        - Reduced positions
        """
        # This would require comparing two quarters of 13F filings
        # Simplified implementation for now
        
        return {
            "investor": investor_key,
            "note": "Full implementation requires parsing multiple 13F filings"
        }
    
    def get_famous_investors_buying(self, ticker: str) -> List[Dict]:
        """
        Check which famous investors have recently bought this stock.
        """
        buyers = []
        
        ownership = self.get_stock_institutional_owners(ticker)
        
        if "famous_investors" in ownership:
            buyers = ownership["famous_investors"]
        
        return buyers
    
    def close(self):
        self.client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


class WhalePortfolioTracker:
    """
    Track crypto whale portfolios.
    """
    
    def __init__(self):
        self.client = httpx.Client(timeout=30.0)
    
    def get_whale_wallets(self, blockchain: str = "ethereum") -> List[Dict]:
        """
        Get known whale wallets.
        
        Note: This uses public blockchain explorers.
        """
        whales = []
        
        if blockchain == "ethereum":
            # Known ETH whale addresses
            known_whales = {
                "0x00000000219ab540356cBB839Cbe05303d7705Fa": "ETH 2.0 Staking Contract",
                "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2": "Wrapped ETH",
                "0xBE0eB53F46cd790Cd13851d5EFf43D12404d33E8": "Binance 7",
                "0x8315177aB297BA92A06054cE80a67Ed4DBd7ed3a": "Arbitrum Bridge",
            }
            
            for address, name in known_whales.items():
                whales.append({
                    "address": address,
                    "name": name,
                    "blockchain": blockchain
                })
        
        return whales
    
    def get_wallet_balance(self, address: str, blockchain: str = "ethereum") -> Dict:
        """
        Get wallet balance from public API.
        """
        try:
            if blockchain == "ethereum":
                # Use Etherscan or similar (requires API key for high rate)
                response = self.client.get(
                    f"https://api.etherscan.io/api",
                    params={
                        "module": "account",
                        "action": "balance",
                        "address": address,
                        "tag": "latest",
                        "apikey": os.getenv("ETHERSCAN_API_KEY", "")
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "1":
                        balance_wei = int(data.get("result", 0))
                        balance_eth = balance_wei / 1e18
                        return {
                            "address": address,
                            "balance_eth": balance_eth,
                            "blockchain": blockchain
                        }
        except Exception as e:
            logger.warning(f"Failed to get wallet balance: {e}")
        
        return {"address": address, "error": "Failed to fetch balance"}
    
    def close(self):
        self.client.close()


def get_investor_summary() -> Dict[str, Any]:
    """Get summary of all tracked famous investors."""
    return {
        "investors": [
            {
                "key": key,
                "name": inv.name,
                "fund": inv.fund_name,
                "strategy": inv.strategy,
                "description": inv.description
            }
            for key, inv in FAMOUS_INVESTORS.items()
        ],
        "total_tracked": len(FAMOUS_INVESTORS)
    }


def find_stocks_owned_by_gurus(tickers: List[str]) -> Dict[str, List[str]]:
    """
    For a list of tickers, find which ones are owned by famous investors.
    """
    tracker = PortfolioTracker()
    results = {}
    
    try:
        for ticker in tickers:
            owners = tracker.get_famous_investors_buying(ticker)
            if owners:
                results[ticker] = [o["investor"] for o in owners]
    finally:
        tracker.close()
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== Famous Investors Tracked ===\n")
    summary = get_investor_summary()
    
    for inv in summary["investors"]:
        print(f"• {inv['name']} ({inv['fund']})")
        print(f"  Strategy: {inv['strategy']}")
        print(f"  {inv['description']}\n")
    
    print(f"Total: {summary['total_tracked']} investors tracked")
    
    # Test institutional holdings
    print("\n=== Testing AAPL Institutional Ownership ===")
    tracker = PortfolioTracker()
    try:
        ownership = tracker.get_stock_institutional_owners("AAPL")
        print(f"Major Holders: {ownership.get('major_holders', {})}")
        print(f"Famous Investors holding AAPL: {len(ownership.get('famous_investors', []))}")
        for inv in ownership.get("famous_investors", []):
            print(f"  - {inv['investor']}: {inv.get('shares', 'N/A')} shares")
    finally:
        tracker.close()
