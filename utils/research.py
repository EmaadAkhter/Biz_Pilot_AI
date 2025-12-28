from utils.llm import call_llm
import requests
import json
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import time


class SearchManager:
    """Manages Google Custom Search API with usage tracking"""

    def __init__(self, api_key: str = None, cx: str = None,
                 usage_file: str = "./search_usage.json"):
        self.api_key = api_key or os.getenv("GOOGLE_SEARCH_API_KEY")
        self.cx = cx or os.getenv("GOOGLE_SEARCH_CX")
        self.usage_file = usage_file
        self.usage_data = self._load_usage()

        if not self.api_key or not self.cx:
            raise ValueError("Google API credentials not found. Set GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_CX")

    def _load_usage(self) -> dict:
        """Load usage tracking data"""
        if os.path.exists(self.usage_file):
            with open(self.usage_file, 'r') as f:
                return json.load(f)
        return {
            "count": 0,
            "date": datetime.now().strftime("%Y-%m-%d")
        }

    def _save_usage(self):
        """Save usage tracking data"""
        with open(self.usage_file, 'w') as f:
            json.dump(self.usage_data, f, indent=2)

    def _reset_if_needed(self):
        """Reset counter if new day"""
        today = datetime.now().strftime("%Y-%m-%d")

        if self.usage_data["date"] != today:
            self.usage_data = {"count": 0, "date": today}
            self._save_usage()

    def _can_search(self) -> bool:
        """Check if under daily limit"""
        return self.usage_data["count"] < 100

    def search(self, query: str, num_results: int = 5) -> List[Dict]:
        """
        Search using Google Custom Search API
        Returns list of search results
        """
        self._reset_if_needed()

        if not self._can_search():
            raise Exception(f"Daily Google search quota exhausted (100/100). Resets at midnight UTC.")

        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.api_key,
                "cx": self.cx,
                "q": query,
                "num": min(num_results, 10)  # Google max is 10 per request
            }

            print(f"  Searching Google (quota: {self.usage_data['count']}/100)...")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("items", []):
                results.append({
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "displayed_link": item.get("displayLink", "")
                })

            # Update usage
            self.usage_data["count"] += 1
            self._save_usage()

            print(f"  ✓ Found {len(results)} results")
            return results

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                raise Exception("Google API rate limit exceeded. Wait before retrying.")
            elif e.response.status_code == 403:
                raise Exception("Google API quota exceeded or invalid credentials.")
            else:
                raise Exception(f"Google API error: {e}")
        except Exception as e:
            raise Exception(f"Search failed: {str(e)}")

    def get_usage_stats(self) -> dict:
        """Get current usage statistics"""
        self._reset_if_needed()
        return {
            "used": self.usage_data["count"],
            "limit": 100,
            "remaining": 100 - self.usage_data["count"],
            "period": "daily",
            "resets": "midnight UTC",
            "date": self.usage_data["date"]
        }


def gather_market_data(idea: str, customer: str, geography: str, level: int,
                       search_manager: SearchManager) -> dict:
    """Gather real market data from web sources"""

    searches_to_perform = []

    # Basic searches for all levels
    searches_to_perform.extend([
        f"{idea} market size {geography}",
        f"{idea} competitors {geography}",
        f"{customer} pain points {idea}"
    ])

    if level >= 2:
        searches_to_perform.extend([
            f"{idea} pricing comparison {geography}",
            f"{idea} customer reviews complaints",
            f"{idea} industry trends 2024 2025",
            f"{customer} behavior {geography}"
        ])

    if level >= 3:
        searches_to_perform.extend([
            f"{idea} market share leaders",
            f"{idea} competitive analysis",
            f"{idea} regulatory landscape {geography}",
            f"{customer} demographics {geography}",
            f"{idea} emerging trends"
        ])

    # Gather search results
    all_results = []
    successful_searches = 0
    failed_searches = 0

    for i, query in enumerate(searches_to_perform):
        try:
            print(f"\n[{i + 1}/{len(searches_to_perform)}] Query: {query}")
            results = search_manager.search(query, num_results=3 if level == 1 else 5)
            all_results.append({
                "query": query,
                "results": results,
                "result_count": len(results),
                "status": "success"
            })
            successful_searches += 1

            # Rate limiting: delay between searches to be respectful
            if i < len(searches_to_perform) - 1:
                time.sleep(1)

        except Exception as e:
            error_msg = str(e)
            print(f"  ✗ Error: {error_msg}")
            all_results.append({
                "query": query,
                "results": [],
                "error": error_msg,
                "status": "failed"
            })
            failed_searches += 1

            # If quota exhausted, stop trying
            if "quota exhausted" in error_msg.lower():
                print(f"\n⚠ Quota exhausted. Stopping further searches.")
                break

    return {
        "searches_attempted": len(searches_to_perform),
        "searches_completed": len(all_results),
        "successful": successful_searches,
        "failed": failed_searches,
        "results": all_results,
        "usage_stats": search_manager.get_usage_stats()
    }


def do_market_research(idea: str, customer: str, geography: str, level: int,
                       search_manager: Optional[SearchManager] = None) -> dict:
    """
    Perform market research at specified depth level with real web data

    Args:
        idea: Product/service idea
        customer: Target customer segment
        geography: Target geography
        level: Research depth (1=quick, 2=medium, 3=deep)
        search_manager: Optional SearchManager instance (will create default if not provided)

    Returns:
        dict with research analysis based on real data
    """

    # Initialize search manager if not provided
    if search_manager is None:
        search_manager = SearchManager()

    # Step 1: Gather real market data
    print(f"\n{'=' * 60}")
    print(f"Starting Level {level} Market Research")
    print(f"Idea: {idea}")
    print(f"Customer: {customer}")
    print(f"Geography: {geography}")
    print(f"{'=' * 60}")

    market_data = gather_market_data(idea, customer, geography, level, search_manager)

    # Check if we got any results
    if market_data["successful"] == 0:
        return {
            "level": level,
            "idea": idea,
            "error": "No search results obtained. Check API quota and credentials.",
            "usage_stats": market_data["usage_stats"]
        }

    # Step 2: Format the data for LLM analysis
    context = "=== WEB SEARCH RESULTS ===\n\n"
    for search in market_data["results"]:
        context += f"Query: {search['query']}\n"
        if search["status"] == "failed":
            context += f"Error: {search.get('error', 'Unknown error')}\n\n"
            continue
        context += f"Found {search['result_count']} results:\n"
        for idx, result in enumerate(search['results'], 1):
            context += f"  [{idx}] {result['title']}\n"
            context += f"      URL: {result['link']}\n"
            context += f"      Snippet: {result['snippet']}\n"
        context += "\n"

    # Step 3: Have LLM analyze the real data
    if level == 1:
        prompt = f"""
You are a market research analyst. Analyze the following REAL market data and provide a QUICK summary:

Idea: {idea}
Target Customer: {customer}
Geography: {geography}

{context}

Based ONLY on the search results above, provide:
1. Estimated market size (cite sources with URLs)
2. Top 3-5 competitors found
3. SWOT summary
4. 2-3 opportunity areas

Keep it concise. Only use information from the search results. Always cite sources.
"""
    elif level == 2:
        prompt = f"""
You are a market research analyst. Analyze the following REAL market data:

Idea: {idea}
Target Customer: {customer}
Geography: {geography}

{context}

Based on the search results, provide:
1. Market size estimates with source citations (include URLs)
2. Detailed competitor analysis (products, pricing, positioning)
3. Customer pain points identified
4. Existing solutions breakdown
5. Demand signals and trends

Cite specific sources with URLs. Stay under 1000 words.
"""
    else:
        prompt = f"""
You are a senior market research analyst. Analyze the following REAL market data:

Idea: {idea}
Target Customer: {customer}
Geography: {geography}

{context}

Provide a comprehensive report with source citations (include URLs):
1. Competitive intelligence analysis
2. Feature gap analysis
3. Market segmentation insights
4. Geographic trends
5. Customer sentiment analysis
6. Competitive forces assessment
7. Marketing mix considerations
8. Executive summary

Be thorough and cite all sources with URLs.
"""

    print(f"\n{'=' * 60}")
    print("Analyzing data with LLM...")
    print(f"{'=' * 60}\n")

    research_text = call_llm(prompt)

    # Print final stats
    stats = market_data["usage_stats"]
    print(f"\n{'=' * 60}")
    print("Research Complete - Summary:")
    print(f"Searches: {market_data['successful']}/{market_data['searches_attempted']} successful")
    print(f"Google API: {stats['used']}/{stats['limit']} queries used today")
    print(f"Remaining: {stats['remaining']} queries")
    print(f"{'=' * 60}\n")

    return {
        "level": level,
        "idea": idea,
        "searches_attempted": market_data["searches_attempted"],
        "searches_successful": market_data["successful"],
        "research": research_text,
        "raw_data": market_data["results"],
        "usage_stats": stats
    }


def do_market_research_cached(idea: str, customer: str, geography: str, level: int,
                              search_manager: Optional[SearchManager] = None,
                              cache_dir: str = "./cache",
                              cache_expiry_hours: int = 24) -> dict:
    """
    Same as do_market_research but with caching
    Useful to avoid hitting rate limits and save API costs

    Args:
        cache_expiry_hours: Cache expires after this many hours (default 24)
    """
    import hashlib

    # Initialize search manager if not provided
    if search_manager is None:
        search_manager = SearchManager()

    # Create cache key
    cache_key = hashlib.md5(
        f"{idea}_{customer}_{geography}_{level}".encode()
    ).hexdigest()
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")

    # Check cache and expiry
    if os.path.exists(cache_file):
        cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        age_hours = (datetime.now() - cache_time).seconds // 3600

        if datetime.now() - cache_time < timedelta(hours=cache_expiry_hours):
            print(f"\n✓ Using cached results (age: {age_hours}h, expires in {cache_expiry_hours - age_hours}h)")
            with open(cache_file, 'r') as f:
                return json.load(f)
        else:
            print(f"\n⚠ Cache expired (age: {age_hours}h)")

    # Perform research
    result = do_market_research(idea, customer, geography, level, search_manager)

    # Save to cache
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"✓ Results cached for {cache_expiry_hours} hours")

    return result


# Usage example
if __name__ == "__main__":
    # Initialize search manager
    search_manager = SearchManager(
        api_key="your_google_key",
        cx="your_google_cx"
    )

    # Check current usage
    stats = search_manager.get_usage_stats()
    print("Current Google API Usage:")
    print(json.dumps(stats, indent=2))

    # Run research with caching
    result = do_market_research_cached(
        idea="AI-powered meal planning app",
        customer="busy professionals aged 25-40",
        geography="United States",
        level=2,
        search_manager=search_manager,
        cache_expiry_hours=24
    )

    print("\n" + "=" * 60)
    print("RESEARCH REPORT:")
    print("=" * 60)
    print(result["research"])