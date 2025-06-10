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

from agno.tools.yfinance import YFinanceTools

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

# # ************* Paths *************
cwd = Path(__file__).parent
# knowledge_dir = cwd.joinpath("knowledge/brainspark/")
# output_dir = cwd.joinpath("output")

# # Create the output directory if it does not exist
# output_dir.mkdir(parents=True, exist_ok=True)
# *******************************
agent_storage_file: str = f"{cwd}/tmp/agents.db"
memory_storage_file: str = f"{cwd}/tmp/memory.db"

memory_db = SqliteMemoryDb(table_name="product_manager_memory", 
                           db_file=memory_storage_file)

# Create Memory object that wraps the database
memory = Memory(db=memory_db)

# Storage
agent_storage = SqliteStorage(table_name="product_manager_agent", db_file=agent_storage_file)


# # Vectordb - Fix URL format and add embedder here
# api_key = os.getenv("QDRANT_API_KEY")
# qdrant_url = os.getenv("QDRANT_URL")
# collection_name = "script_writer_knowledge"

# vector_db = Qdrant(
#     collection=collection_name,
#     url=qdrant_url,
#     api_key=api_key,
#     embedder=GeminiEmbedder(id="text-embedding-004", 
#                             dimensions=768, 
#                             api_key=os.getenv("4DCNNGEMINI")),
# )


# # Configure Gemini models for chunking
# chunking_model = Gemini(id="gemini-2.0-flash-lite", temperature=0.2, api_key=os.getenv("3DCNNGEMINI"))

# # Knowledge Base - Configure with Gemini chunking
# pdf_knowledge_base = PDFKnowledgeBase(
#     path="/home/z4hid/Desktop/githubProjects/brainspark_agentic_workflow/knowledge/brainspark/brainspark.pdf",
#     vector_db=vector_db,
#     # reader=PDFReader(chunk=True, chunk_size=5000),
#     chunking_strategy=AgenticChunking(model=chunking_model)
# )


# Agents
product_manager = Agent(
    name="Product Manager",
    agent_id="product_manager",
    model=Gemini(id="gemini-2.0-flash", temperature=0.2, api_key=os.getenv("5DCNNGEMINI")),
    description="""
    The Product Manager agent plays a strategic role in defining, refining, and managing BrainSpark Digital's suite of service offerings, 
    which include Web Development, SEO, AI, and Graphic Design. Its purpose is to ensure these services continuously meet evolving market 
    needs, maintain a competitive edge, and align with the company's overall strategic and growth objectives. This agent appears to focus
    on the "Strategy" output that directly influences "Growth" initiatives.
""",
    instructions="""
    Conduct a thorough analysis of BrainSpark Digital's current service offerings (Web Development, SEO, AI, Graphic Design) in the context of prevailing market trends, technological advancements, and the competitive landscape, particularly focusing on opportunities and challenges for a modern digital agency.

    Identify and evaluate opportunities for launching new service packages or refining existing ones. Pay particular attention to 'productizing' services—creating standardized packages with clear deliverables and pricing—to enhance scalability and marketability.

    Define clear and compelling value propositions for each core service and any new proposed packages. Ensure these value propositions directly address identified client pain points and align with the overarching Brandscript narrative, if available.

    Undertake 'Service-Market Fit Validation' analysis. This may involve proposing specific market testing experiments, analyzing client feedback on current offerings, or surveying potential customers to gauge demand and perceived value for new or modified services.

    Propose and design methodologies for 'A/B Testing of Service Packages and Pricing Tiers'. The goal is to empirically determine optimal configurations that maximize both revenue generation and perceived client value.

    Develop and refine strategies for a 'Transparent and Structured Client Onboarding' process. This aims to enhance the initial client experience, set clear expectations, and improve the Activation stage of the customer lifecycle.

    Continuously evaluate the 'Scalability for Service Delivery' for all services. Propose process improvements, automation opportunities (potentially leveraging BrainSpark's own AI capabilities), or resource allocation adjustments to ensure services can be delivered efficiently and effectively as client volume grows.

    Collaborate closely with the Growth Hacker agent on strategies and experiments related to the 'Revenue' stage of the AARRR funnel, ensuring service offerings are structured to support monetization goals.

    Use the following tools to get the latest information:
    - Google Search
    - Google Trends
    - Tavily
    - DuckDuckGo
    - YFinance 
    - Wikipedia
""",
    memory=memory,
    enable_user_memories=True,
    storage=agent_storage,
    add_history_to_messages=True,
    add_datetime_to_instructions=True,
    user_id="z4hid",
    debug_mode=True,
    markdown=True,
    role="product_manager",
    tools=[
        GoogleSearchTools(fixed_max_results=15),
        DuckDuckGoTools(fixed_max_results=10),
        TavilyTools(),
        FirecrawlTools(),
        YFinanceTools(),
        WikipediaTools(),
    ],
    show_tool_calls=True
)

if __name__ == "__main__":
    try:
        # pdf_knowledge_base.load(recreate=False)
        # Comment out after first run
        
        product_manager.print_response("Competitor analysis reports detailing services, pricing, and positioning of other digital agencies", stream=True)
    except Exception as e:
        print(f"Error: {e}")
