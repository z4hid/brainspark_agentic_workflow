from pathlib import Path
from textwrap import dedent

from agno.agent import Agent

from agno.models.google import Gemini
from agno.embedder.google import GeminiEmbedder

from agno.storage.sqlite import SqliteStorage

from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.memory.v2.memory import Memory

from agno.vectordb.qdrant import Qdrant
from agno.vectordb.search import SearchType

from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.tavily import TavilyTools
from agno.tools.wikipedia import WikipediaTools
from agno.tools.exa import ExaTools

from agno.tools.website import WebsiteTools
from agno.tools.firecrawl import FirecrawlTools

from agno.tools.csv_toolkit import CsvTools
from agno.tools.pandas import PandasTools

from agno.tools.file import FileTools
from agno.tools.python import PythonTools
from agno.tools.shell import ShellTools

from agno.knowledge.combined import CombinedKnowledgeBase
from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.knowledge.csv import CSVKnowledgeBase, CSVReader
from agno.document.chunking.agentic import AgenticChunking

import os

from dotenv import load_dotenv
load_dotenv()

# ************* Paths *************
cwd = Path(__file__).parent
knowledge_dir = cwd.joinpath("knowledge/lean/")
output_dir = cwd.joinpath("output")

# # Create the output directory if it does not exist
# output_dir.mkdir(parents=True, exist_ok=True)
# *******************************
agent_storage_file: str = "tmp/agents.db"
memory_storage_file: str = "tmp/memory.db"

memory_db = SqliteMemoryDb(table_name="seo_specialist_memory", 
                           db_file=memory_storage_file)

# Create Memory object that wraps the database
memory = Memory(db=memory_db)

# Storage
agent_storage = SqliteStorage(table_name="seo_specialist_agent", db_file=agent_storage_file)


# Vectordb - Fix URL format and add embedder here
api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
collection_name = "seo_specialist_knowledge"
keyword_collections = "keyword_collections"
info_collections = "seo_info_collections"


vector_db = Qdrant(
    collection=info_collections,
    url=qdrant_url,
    api_key=api_key,
    embedder=GeminiEmbedder(id="text-embedding-004", 
                            dimensions=768, 
                            api_key=os.getenv("2DCNNGEMINI")),
)

keyword_vector_db = Qdrant(
    collection=keyword_collections,
    url=qdrant_url,
    api_key=api_key,
    embedder=GeminiEmbedder(id="text-embedding-004", 
                            dimensions=768, 
                            api_key=os.getenv("1DCNNGEMINI")),
)

combined_vector_db = Qdrant(
    collection=collection_name,
    url=qdrant_url,
    api_key=api_key,
    embedder=GeminiEmbedder(id="text-embedding-004", 
                            dimensions=768, 
                            api_key=os.getenv("1DCNNGEMINI")),
)

# Configure Gemini models for chunking
chunking_model = Gemini(id="gemini-2.0-flash-lite", temperature=0.2, api_key=os.getenv("2DCNNGEMINI"))

# Knowledge Base - Configure with Gemini chunking
pdf_knowledge_base = PDFKnowledgeBase(
    path="/home/z4hidhasan/Desktop/z4hid/github/brainspark_agentic_workflow/knowledge/lean/lean_seo.pdf",
    #path=knowledge_dir.joinpath("lean_seo.pdf"),
    vector_db=vector_db,
    reader=PDFReader(chunk=True, chunk_size=5000),
    chunking_strategy=AgenticChunking(model=chunking_model)
)

csv_knowledge_base = CSVKnowledgeBase(
    #path=knowledge_dir.joinpath("webdata.csv"),
    path="/home/z4hidhasan/Desktop/z4hid/github/brainspark_agentic_workflow/knowledge/lean/",
    vector_db=vector_db,
    # chunking_strategy=AgenticChunking(model=chunking_model)
)

combined_knowledge_base = CombinedKnowledgeBase(
    sources=[pdf_knowledge_base, csv_knowledge_base],
    vector_db=keyword_vector_db,
    chunking_strategy=AgenticChunking(model=chunking_model)
)

# Agents
seo_specialist = Agent(
    name="SEO Specialist",
    agent_id="seo_specialist",
    model=Gemini(id="gemini-2.0-flash", temperature=0.2, api_key=os.getenv("2DCNNGEMINI")),
    description="""
    You are an expert SEO Specialist focused on implementing Lean SEO strategies. Your purpose is to conduct comprehensive keyword research,
    analyze existing SEO performance, identify strategic opportunities, and provide actionable recommendations to enhance organic search visibility. 
    Operating within the Lean SEO framework, you emphasize agile methodologies and data-driven decision making to maximize SEO impact efficiently.

    Always ask multiple questions to the Knowledge Base to understand:
    - Current SEO performance metrics and benchmarks
    - Target keyword opportunities and search intent
    - Technical SEO issues and optimization priorities
    - Content gaps and optimization recommendations
    - Competitor SEO strategies and market positioning
    - Implementation of Lean SEO principles and best practices
    
    Your output must be in Markdown format with clear, actionable insights and recommendations.
""",
    instructions="""
    1. Conduct Lean Keyword Research:
        - Identify seed keywords based on service offerings and Brandscript
        - Expand keyword list using research tools
        - Filter based on:
            * Service relevance
            * Realistic search volume for Worldwide market
            * Competition levels (focus on low-competition opportunities)
            * Clear searcher intent (informational/navigational/commercial/transactional)
        - Use Google Search and DuckDuckGo to validate search intent
        - Leverage Tavily for market research and trend analysis

    2. Develop Keyword Clusters for Minimum Viable Content (MVC):
        - Create clusters with primary target keywords
        - Include related secondary and long-tail keywords
        - Define clear MVC Hypothesis for each cluster
        - Set measurable outcome expectations and timeframes
        - Use CSV Tools to organize and analyze keyword data
        - Utilize Exa for competitive keyword gap analysis

    3. Analyze Current Online Presence:
        - Evaluate website structure and navigation
        - Assess UX factors
        - Review on-page SEO elements
        - Check technical SEO health:
            * Robots.txt and sitemap.xml
            * Mobile-friendliness
            * Page speed metrics
        - Identify quick-win technical fixes
        - Use FirecrawlTools to crawl the website and analyze the technical SEO issues

    4. Develop On-Page Optimization Strategies:
        - URL structure optimization
        - Title tags and meta descriptions
        - Heading hierarchy (H1-H6)
        - Content structure and readability
        - Strategic keyword placement
        - Internal/external linking practices
        - Use FirecrawlTools to crawl the website and analyze the on-page SEO issues

    5. Plan Lean Link-Building Experiments:
        - Build local citations for Gazipur, Dhaka, Bangladesh office
        - Monitor and respond to HARO queries
        - Conduct broken link building campaigns
        - Pursue strategic guest posting opportunities

    6. Implement Performance Monitoring:
        - Track MVCs and SEO experiments over 30-day cycles
        - Monitor key metrics:
            * Organic traffic
            * Keyword rankings
            * Impressions
            * Click-through rates
            * Conversion actions

    7. Analyze and Iterate:
        - Categorize content performance:
            * High Performer
            * Promising
            * Underperforming
            * Inconclusive
        - Make data-driven decisions to:
            * Scale successful initiatives
            * Optimize promising content
            * Pivot/abandon underperforming efforts
        - Document all learnings and insights
        - Use CSV Tools or PandasTools and Exa for comprehensive performance analysis
""",
    memory=memory,
    enable_user_memories=True,
    knowledge=combined_knowledge_base,
    search_knowledge=True,
    add_references=True,
    enable_agentic_knowledge_filters=True,
    storage=agent_storage,
    add_history_to_messages=True,
    add_datetime_to_instructions=True,
    user_id="z4hid",
    debug_mode=True,
    markdown=True,
    role="seo_specialist",
    tools=[
        GoogleSearchTools(fixed_max_results=15),
        DuckDuckGoTools(fixed_max_results=10),
        TavilyTools(),
        WikipediaTools(),
        ExaTools(),
        #CsvTools(csvs=["/home/z4hidhasan/Desktop/z4hid/github/brainspark_agentic_workflow/knowledge/lean/webdata.csv"]),
        CsvTools(),
        FirecrawlTools(),
        PandasTools(),
    ],
    show_tool_calls=True
)

if __name__ == "__main__":
    try:
        BANDSCRIPT = """
Master BrandScript: BrainSpark Digital - AI Analytics for Small Businesses                                                 ┃
┃                                                                                                                                                                           ┃
┃ 1. The Character:                                                                                                                                                         ┃
┃                                                                                                                                                                           ┃
┃  • Who: Small business owners (e.g., retail shops, restaurants, professional services, online stores)                                                                     ┃
┃  • What they want: To grow their business, make informed decisions, and compete effectively without being overwhelmed by data. They want clarity, control, and confidence ┃
┃    in their business strategy.                                                                                                                                            ┃
┃                                                                                                                                                                           ┃
┃ 2. The Problem:                                                                                                                                                           ┃
┃                                                                                                                                                                           ┃
┃  • External:                                                                                                                                                              ┃
┃     • Lack of actionable insights from business data.                                                                                                                     ┃
┃     • Difficulty understanding complex analytics tools.                                                                                                                   ┃
┃     • Spending too much time on manual data collection and reporting.                                                                                                     ┃
┃     • Inability to identify trends and opportunities.                                                                                                                     ┃
┃  • Internal:                                                                                                                                                              ┃
┃     • Frustration with feeling "lost in the data."                                                                                                                        ┃
┃     • Fear of making wrong decisions based on gut feelings instead of facts.                                                                                              ┃
┃     • Overwhelm and anxiety about keeping up with competitors.                                                                                                            ┃
┃     • Self-doubt about their ability to understand and use data effectively.                                                                                              ┃
┃  • Philosophical:                                                                                                                                                         ┃
┃     • It's unfair that large corporations have access to sophisticated analytics while small businesses are left behind.                                                  ┃
┃     • Small businesses deserve the same data-driven advantages as larger companies.                                                                                       ┃
┃     • Business decisions should be based on facts, not guesswork.                                                                                                         ┃
┃  • The Villain: Data Overwhelm & Uncertainty. This manifests as:                                                                                                          ┃
┃     • Complexity: Confusing dashboards and jargon-filled reports.                                                                                                         ┃
┃     • Inaction: Paralysis caused by too much information and not enough clarity.                                                                                          ┃
┃     • Missed Opportunities: Failure to identify and capitalize on emerging trends.                                                                                        ┃
┃                                                                                                                                                                           ┃
┃ 3. The Guide: BrainSpark Digital                                                                                                                                          ┃
┃                                                                                                                                                                           ┃
┃  • Empathy:                                                                                                                                                               ┃
┃     • "We understand that as a small business owner, you're already wearing many hats. You don't have time to become a data scientist."                                   ┃
┃     • "We know how frustrating it is to feel like you're missing out on opportunities because you can't make sense of your data."                                         ┃
┃     • "We get that you're passionate about your business, and you want to make informed decisions without getting bogged down in technical details."                      ┃
┃  • Authority:                                                                                                                                                             ┃
┃     • "BrainSpark Digital specializes in AI-powered analytics solutions designed specifically for small businesses."                                                      ┃
┃     • "Our platform transforms your raw data into clear, actionable insights that drive growth."                                                                          ┃
┃     • "We have a proven track record of helping small businesses like yours increase revenue, improve efficiency, and gain a competitive edge."                           ┃
┃     • "Our AI algorithms are constantly learning and adapting to provide you with the most relevant and up-to-date information."                                          ┃
┃                                                                                                                                                                           ┃
┃ 4. The Plan:                                                                                                                                                              ┃
┃                                                                                                                                                                           ┃
┃  1 Free Consultation: Schedule a free consultation to discuss your business goals and data challenges.                                                                    ┃
┃  2 Data Integration: We'll seamlessly integrate our AI analytics platform with your existing data sources (e.g., POS, CRM, website).                                      ┃
┃  3 Personalized Insights: Receive customized reports and dashboards that highlight key trends, opportunities, and areas for improvement.                                  ┃
┃  4 Ongoing Support: Benefit from ongoing support and training to ensure you're getting the most out of our platform.                                                      ┃
┃                                                                                                                                                                           ┃
┃ 5. Calls to Action:                                                                                                                                                       ┃
┃                                                                                                                                                                           ┃
┃  • Direct CTA: "Start Your Free Trial Today" or "Get a Free Data Assessment"                                                                                              ┃
┃  • Transitional CTA: "Download our Free Guide: 5 Data-Driven Strategies to Grow Your Small Business" or "Watch a Demo"                                                    ┃
┃                                                                                                                                                                           ┃
┃ 6. Failure (What's Avoided):                                                                                                                                              ┃
┃                                                                                                                                                                           ┃
┃  • Losing customers to competitors who are using data more effectively.                                                                                                   ┃
┃  • Wasting time and money on marketing campaigns that don't deliver results.                                                                                              ┃
┃  • Making critical business decisions based on guesswork and intuition.                                                                                                   ┃
┃  • Missing out on opportunities for growth and expansion.                                                                                                                 ┃
┃  • Feeling overwhelmed and stressed by the constant pressure to keep up.                                                                                                  ┃
┃                                                                                                                                                                           ┃
┃ 7. Success (The Desired Outcome):                                                                                                                                         ┃
┃                                                                                                                                                                           ┃
┃  • Increased Revenue: See a measurable increase in sales and profitability.                                                                                               ┃
┃  • Improved Efficiency: Streamline your operations and reduce wasted resources.                                                                                           ┃
┃  • Data-Driven Decisions: Make confident, informed decisions based on clear insights.                                                                                     ┃
┃  • Competitive Advantage: Gain a competitive edge by identifying and capitalizing on emerging trends.                                                                     ┃
┃  • Peace of Mind: Feel confident and in control of your business destiny.                                                                                                 ┃
┃  • Character Transformation: From overwhelmed and uncertain to empowered and strategic. The small business owner transforms into a confident data-driven leader.          ┃
┃                                                                                                                                                                           ┃
┃ One-Liner: BrainSpark Digital: AI-powered analytics that transforms your small business data into big results.                                                            ┃
┃                                                                                                                                                                           ┃
┃ Controlling Idea: By providing small businesses with accessible and actionable AI-driven analytics, BrainSpark Digital empowers them to overcome data overwhelm, make     ┃
┃ informed decisions, and achieve sustainable growth, leveling the playing field and ensuring their success in an increasingly competitive market.
"""
        # Comment out after first run
        combined_knowledge_base.load(recreate=False)
        
        seo_specialist.print_response("DO an extensive keyword research and develop keyword clusters for the following keywords: Website Development, AI, AI Agents " + BANDSCRIPT, stream=True)
    except Exception as e:
        print(f"Error: {e}")
