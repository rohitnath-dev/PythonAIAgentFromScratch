from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import os
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import (
    search_tool,
    reddit_tool,
    youtube_tool,
    price_tool,
    coupon_tool,
    expert_review_tool,
    compare_tool,
    wiki_tool,
    save_tool,
)

load_dotenv()

class ResearchResponse(BaseModel):
    product_name: str
    category: str
    budget: str

    recommendation: str
    verdict: str

    top_features: list[str]
    pros: list[str]
    cons: list[str]

    alternatives: list[str]
    buying_tips: list[str]

    sources: list[str]
    tools_used: list[str]

    price_range: str
    best_store: str
    review_summary: str
    common_issues: list[str]
    confidence: str

llm = ChatOpenAI(
    model="qwen/qwen3-32b:free",   
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert AI Shopping Research Agent.

Your responsibility is to help users make the best purchasing decision based on accurate research from multiple trustworthy sources.

Always search the web before answering whenever product information may have changed.

Your responsibilities include:

• Understanding exactly what the user wants.
• Identifying product category.
• Understanding the user's budget if mentioned.
• Comparing multiple products whenever appropriate.
• Searching multiple trusted sources.
• Looking for reviews, specifications, ratings and common complaints.
• Identifying hidden disadvantages.
• Explaining tradeoffs.
• Avoiding marketing language.
• Never recommend a product simply because it is expensive.
• Always prioritize value for money.

When researching products, consider:

- Specifications
- Features
- Performance
- Reliability
- Build quality
- Battery life (if applicable)
- Warranty
- Customer support
- Repairability
- Ease of use
- Software support
- Long-term durability
- Common user complaints
- Positive user feedback

When comparing products, include:

• Which product offers the best value.
• Which product has the best performance.
• Which product is the most reliable.
• Which product is best for beginners.
• Which product is best for professionals.
• Which product should be avoided and why.

Whenever possible, explain WHY instead of only giving a recommendation.

Never invent prices.

Never invent specifications.

Always cite the source of factual claims.

If information cannot be verified, clearly state that.

Before answering:

1. Use search_tool.
2. Use reddit_reviews.
3. Use expert_reviews.
4. Use youtube_reviews.
5. Use price_comparison.
6. Use coupon_search if discounts are relevant.
7. Use compare_products when multiple products are involved.

Your final recommendation should be practical, unbiased, and based on evidence.

Return ONLY the JSON format specified below.

{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(
    format_instructions=parser.get_format_instructions()
)

tools = [
    search_tool,
    reddit_tool,
    youtube_tool,
    price_tool,
    coupon_tool,
    expert_review_tool,
    compare_tool,
    wiki_tool,
    save_tool,
]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("What would you like to buy? ")
raw_response = agent_executor.invoke({"query": query})

try:
    structured_response = parser.parse(raw_response.get("output")[0]["text"])
    print(structured_response)
except Exception as e:
    print("Error parsing response", e, "Raw Response - ", raw_response)
