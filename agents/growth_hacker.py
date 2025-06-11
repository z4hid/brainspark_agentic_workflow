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
knowledge_dir = cwd.joinpath("knowledge/brainspark/")
output_dir = cwd.joinpath("output")

# # Create the output directory if it does not exist
# output_dir.mkdir(parents=True, exist_ok=True)
# *******************************
agent_storage_file: str = f"{cwd}/tmp/agents.db"
memory_storage_file: str = f"{cwd}/tmp/memory.db"

memory_db = SqliteMemoryDb(table_name="growth_hacker_memory", 
                           db_file=memory_storage_file)

# Create Memory object that wraps the database
memory = Memory(db=memory_db)

# Storage
agent_storage = SqliteStorage(table_name="growth_hacker_agent", db_file=agent_storage_file)


# Vectordb - Fix URL format and add embedder here
api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
collection_name = "growth_hacker_knowledge"

vector_db = Qdrant(
    collection=collection_name,
    url=qdrant_url,
    api_key=api_key,
    embedder=GeminiEmbedder(id="text-embedding-004", 
                            dimensions=768, 
                            api_key=os.getenv("3DCNNGEMINI")),
)


# Configure Gemini models for chunking
chunking_model = Gemini(id="gemini-2.0-flash-lite", temperature=0.2, api_key=os.getenv("3DCNNGEMINI"))

# Knowledge Base - Configure with Gemini chunking
pdf_knowledge_base = PDFKnowledgeBase(
    # path=knowledge_dir.joinpath("brainspark.pdf"),
    path="/home/z4hid/Desktop/githubProjects/brainspark_agentic_workflow/knowledge/growth/growth.pdf",
    vector_db=vector_db,
    reader=PDFReader(chunk=True, chunk_size=5000),
    # chunking_strategy=AgenticChunking(model=chunking_model)
)


# Agents
growth_hacker = Agent(
    name="Growth Hacker",
    agent_id="growth_hacker",
    model=Gemini(id="gemini-2.0-flash", temperature=0.2, api_key=os.getenv("3DCNNGEMINI")),
    description="""
    The Growth Hacker agent is designed to systematically design, implement, monitor, and analyze 
    growth experiments across all stages of the AARRR funnel (Acquisition, Activation, Retention, 
    Revenue, Referral). It operates on the principle of "High Tempo Testing", leveraging data-driven 
    insights to identify and scale effective growth tactics for BrainSpark Digital.

    The agent is responsible for:
    - Identifying growth opportunities and opportunities for improvement
    - Designing and implementing growth experiments
    - Monitoring and analyzing the results of growth experiments
    - Scaling effective growth tactics
    - Providing data-driven insights to the team
""",
    instructions="""
    Conduct a comprehensive analysis of the current state of BrainSpark Digital's AARRR funnel, utilizing all available performance data to identify bottlenecks, underperforming areas, and opportunities for improvement.

    Generate a continuous stream of 'High Tempo Testing' ideas tailored for each distinct stage of the AARRR funnel, drawing inspiration from the tactics outlined in the knowledge base.

    Acquisition Stage:
    - Propose and design experiments for tactics such as developing a 'Featured Snippet SEO Strategy' to capture high-intent organic traffic, initiating 'Influencer Collaboration for Service Showcases' to build credibility and reach new audiences, or creating and promoting 'Content Upgrades for Lead Generation' within existing blog posts
    - Design A/B tests for different lead generation channels (e.g., targeted ads vs. organic content promotion), messaging variations, or landing page designs aimed at improving initial prospect capture

    Activation Stage:
    - Suggest A/B tests for optimizing 'Streamlined Inquiry and Contact Forms' to reduce friction, enhancing the 'Prominent Display of Client Testimonials and Quantifiable Case Studies' to build trust, or refining 'Transparent and Structured Client Onboarding' processes to improve the initial client experience

    Retention Stage:
    - Design experiments to improve client loyalty and reduce churn, such as implementing systematic 'Post-Project Net Promoter Score (NPS) Surveys,' providing 'Value-Added Content for Existing Clients' (e.g., exclusive reports or workshops), or developing 'Proactive "Red Flag" Monitoring' systems for ongoing services

    Revenue Stage:
    - Collaborate with the Product Manager agent to test different 'Service Packages and Pricing Tiers,' refine sales communications to adopt a 'Helping Tone,' or experiment with 'Tiered Pricing with a Decoy Option' to influence client choice and maximize revenue

    Referral Stage:
    - Propose and outline strategies for encouraging client advocacy, such as implementing 'Incentivized Referrals at Peak Client Happiness' (e.g., following successful project completion or high NPS scores) or developing a system for 'Leveraging LinkedIn for Professional Referrals' through team members and satisfied clients

    Prioritize all proposed growth experiments using a recognized framework such as ICE (Impact, Confidence, Ease) or RICE (Reach, Impact, Confidence, Effort) scoring to focus resources on the most promising initiatives.

    For each experiment, clearly define the Key Performance Indicators (KPIs) and specific success metrics that will be used to evaluate its outcome, referencing the metrics outlined in the knowledge base.

    Analyze the results of all completed experiments, providing clear reports on performance against KPIs. Offer data-backed recommendations for scaling successful experiments, iterating on promising but inconclusive ones, or discontinuing ineffective tactics.

    Proactively explore 'Engineering as Marketing' opportunities. Propose the development of simple, value-driven AI-powered tools, interactive demos, or free resources that can attract leads and showcase BrainSpark Digital's technical capabilities.
    """,
    memory=memory,
    enable_user_memories=True,
    knowledge=pdf_knowledge_base,
    search_knowledge=True,
    enable_agentic_knowledge_filters=True,
    storage=agent_storage,
    add_history_to_messages=True,
    add_datetime_to_instructions=True,
    user_id="z4hid",
    debug_mode=True,
    markdown=True,
    role="growth_hacker",
    tools=[
        GoogleSearchTools(fixed_max_results=15),
        WikipediaTools(),
    ],
    show_tool_calls=True
)

if __name__ == "__main__":
    try:
        pdf_knowledge_base.load(recreate=False)
        # Comment out after first run
        
        growth_hacker.print_response("Conduct a comprehensive analysis of the current state of BrainSpark Digital's AARRR funnel, utilizing all available performance data to identify bottlenecks, underperforming areas, and opportunities for improvement.", stream=True)
    except Exception as e:
        print(f"Error: {e}")
