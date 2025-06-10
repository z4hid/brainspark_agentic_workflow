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

# # ************* Paths *************
# cwd = Path(__file__).parent
# knowledge_dir = cwd.joinpath("knowledge/brainspark/")
# output_dir = cwd.joinpath("output")

# # Create the output directory if it does not exist
# output_dir.mkdir(parents=True, exist_ok=True)
# *******************************
agent_storage_file: str = "tmp/agents.db"
memory_storage_file: str = "tmp/memory.db"

memory_db = SqliteMemoryDb(table_name="social_media_manager_memory", 
                           db_file=memory_storage_file)

# Create Memory object that wraps the database
memory = Memory(db=memory_db)

# Storage
agent_storage = SqliteStorage(table_name="social_media_manager_agent", db_file=agent_storage_file)


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
social_media_manager = Agent(
    name="Social Media Manager",
    agent_id="social_media_manager",
    model=Gemini(id="gemini-2.0-flash", temperature=0.2, api_key=os.getenv("5DCNNGEMINI")),
    description="""
   The Social Media Manager agent is responsible for creating, curating, and managing social media posts across various platforms. 
   It aims to adapt core content and brand messages for optimal social engagement, interact with the online community, and drive 
   traffic to BrainSpark Digital's primary digital assets.
""",
    instructions="""
    Adapt approved long-form content (such as blog posts, articles, service details, or video summaries) into concise, engaging, and platform-specific social media posts suitable for dissemination on channels like LinkedIn, Twitter, Facebook, Instagram, etc.

    Craft compelling and contextually appropriate captions, questions, and calls to action designed to resonate with the specific audience and format of each social media platform.

    Search for the latest trends in AI using Google Search, Google Trends, Tavily and DuckDuckGo. Scrape the sites using Firecrawl. use these tools for accurate and up to date information.

    Incorporate relevant hashtags and keywords into social media copy. These can be informed by insights from the SEO Specialist, analysis of trending topics using Google Trends, or platform-specific research.

    If scheduling capabilities are integrated or if instructed as part of a campaign, propose optimal posting times for different platforms to maximize reach and engagement (this may require access to platform analytics or general best practices).

    Draft thoughtful and brand-aligned responses to common comments, questions, or inquiries received on social media posts, based on approved messaging, the Brandscript, and information from the knowledge base.

    Conceptualize or describe visuals that should accompany social media posts, aligning with the idea of leveraging 'shareable infographics and visually appealing content'. This might involve generating prompts for an image generation AI or describing requirements for a human graphic designer.

    Develop social media posts that effectively promote 'Content Upgrades' or lead magnets (such as downloadable guides, checklists, or whitepapers) offered by BrainSpark Digital, aiming to capture leads from social channels.

    Create social media content that shares client success stories, positive testimonials, and quantifiable results, framing them appropriately for social media consumption and ensuring client permissions are respected.

    Use the following tools to get the latest information:
    - Google Search
    - Google Trends
    - Tavily
    - DuckDuckGo
""",
    memory=memory,
    enable_user_memories=True,
    storage=agent_storage,
    add_history_to_messages=True,
    add_datetime_to_instructions=True,
    user_id="z4hid",
    debug_mode=True,
    markdown=True,
    role="social_media_manager",
    tools=[
        GoogleSearchTools(fixed_max_results=15),
        DuckDuckGoTools(fixed_max_results=10),
        TavilyTools(),
        FirecrawlTools(),
    ],
    show_tool_calls=True
)

if __name__ == "__main__":
    try:
        # pdf_knowledge_base.load(recreate=False)
        # Comment out after first run
        
        social_media_manager.print_response("Write a social media post for LinkedIn about the latest trends in AI", stream=True)
    except Exception as e:
        print(f"Error: {e}")
