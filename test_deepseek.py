#!/usr/bin/env python3
"""Test Deepseek API integration."""

from dotenv import load_dotenv
load_dotenv()

from trader.analysis.ai_analyzer import AIAnalyzer
import httpx

print("=== Available AI Providers ===")
providers = AIAnalyzer.get_available_providers()
for name, info in providers.items():
    status = "✅ READY" if info["has_api_key"] else "❌ No API Key"
    print(f"{name}: {status} - {info['model']}")

print()
print("=== Testing Deepseek API ===")

try:
    analyzer = AIAnalyzer(provider='deepseek')
    
    # Test 1: Sentiment Analysis
    print("\n1. Testing Sentiment Analysis...")
    result = analyzer.analyze_sentiment("Apple reports record Q4 earnings beating analyst expectations by 15%")
    print(f"   Sentiment: {result.sentiment.name}")
    print(f"   Score: {result.score}")
    print(f"   Confidence: {result.confidence}")
    print(f"   Reasoning: {result.reasoning[:150]}...")
    
    # Test 2: Polish market news
    print("\n2. Testing Polish Market News Analysis...")
    result2 = analyzer.analyze_sentiment("KGHM osiągnęło rekordowe zyski dzięki wysokim cenom miedzi na światowych rynkach")
    print(f"   Sentiment: {result2.sentiment.name}")
    print(f"   Score: {result2.score}")
    print(f"   Confidence: {result2.confidence}")
    print(f"   Reasoning: {result2.reasoning[:150]}...")
    
    # Test 3: Company Evaluation
    print("\n3. Testing Company Evaluation...")
    fundamentals = {
        "symbol": "AAPL",
        "name": "Apple Inc.",
        "pe_ratio": 28.5,
        "market_cap": 3000000000000,
        "revenue_growth": 0.12,
        "profit_margin": 0.25,
        "debt_to_equity": 1.8,
        "dividend_yield": 0.005
    }
    evaluation = analyzer.evaluate_company(fundamentals)
    print(f"   Recommendation: {evaluation.recommendation}")
    print(f"   Score: {evaluation.overall_score}/10")
    print(f"   Strengths: {evaluation.strengths[:2]}")
    print(f"   Weaknesses: {evaluation.weaknesses[:2]}")
    
    analyzer.close()
    print("\n✅ All Deepseek API tests PASSED!")

except httpx.HTTPStatusError as e:
    if e.response.status_code == 402:
        print(f"\n⚠️ Deepseek API - Payment Required (402)")
        print("   Your Deepseek account needs to be topped up with credits.")
        print("   Visit https://platform.deepseek.com/ to add funds.")
        print("\n   ✅ API Configuration is CORRECT - just needs credits!")
    elif e.response.status_code == 401:
        print(f"\n❌ Authentication failed (401)")
        print("   Check your DEEPSEEK_API_KEY in .env file")
    else:
        print(f"\n❌ HTTP Error: {e.response.status_code}")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
