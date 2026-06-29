from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime

search = DuckDuckGoSearchRun()

search_tool = Tool(
    name="search",
    func=search.run,
    description="Search the web for general product information.",
)

api_wrapper = WikipediaAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=300,
)

wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

def save_to_txt(data: str, filename: str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    
    return f"Data successfully saved to {filename}"

save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves structured research data to a text file.",
)


def reddit_reviews(query: str):
    return search.run(f"site:reddit.com {query}")

reddit_tool = Tool(
    name="reddit_reviews",
    func=reddit_reviews,
    description="Find Reddit discussions and user experiences."
)


def youtube_reviews(query: str):
    return search.run(
        f"site:youtube.com {query} review OR unboxing OR camera test OR battery test OR gaming"
    )

youtube_tool = Tool(
    name="youtube_reviews",
    func=youtube_reviews,
    description=(
        "Search YouTube for product reviews, unboxing videos, "
        "camera tests, battery tests and long-term user experiences."
    ),
)


def price_comparison(query: str):
    search_query = f"""
    {query} price Amazon India
    OR Flipkart
    OR Croma
    OR Reliance Digital
    OR Vijay Sales
    OR official store
    """

    return search.run(search_query)

price_tool = Tool(
    name="price_comparison",
    func=price_comparison,
    description=(
        "Compare product prices across Amazon, Flipkart, Croma, "
        "Reliance Digital, Vijay Sales and official stores."
    ),
)


def coupon_search(query: str):
    return search.run(
        f"{query} coupon OR promo code OR bank offer OR cashback OR exchange offer OR discount India"
    )

coupon_tool = Tool(
    name="coupon_search",
    func=coupon_search,
    description=(
        "Find coupons, promo codes, cashback offers, exchange offers "
        "and bank discounts for a product."
    ),
)


def expert_reviews(query: str):
    return search.run(
        f"""
        {query}
        site:gsmarena.com
        OR site:rtings.com
        OR site:notebookcheck.net
        OR site:techradar.com
        OR site:tomsguide.com
        OR site:androidauthority.com
        OR site:91mobiles.com
        review
        """
    )

expert_review_tool = Tool(
    name="expert_reviews",
    func=expert_reviews,
    description=(
        "Search trusted expert review websites for detailed product "
        "reviews, benchmarks, ratings and long-term analysis."
    ),
)


def compare_products(query: str):
    return search.run(
        f"""
        {query}
        comparison
        vs
        benchmark
        pros
        cons
        which is better
        """
    )

compare_tool = Tool(
    name="compare_products",
    func=compare_products,
    description=(
        "Compare two or more products using benchmarks, features, "
        "performance, pros, cons and value for money."
    ),
)
