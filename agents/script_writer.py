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
agent_storage_file: str = "tmp/agents.db"
memory_storage_file: str = "tmp/memory.db"

memory_db = SqliteMemoryDb(table_name="script_writer_memory", 
                           db_file=memory_storage_file)

# Create Memory object that wraps the database
memory = Memory(db=memory_db)

# Storage
agent_storage = SqliteStorage(table_name="script_writer_agent", db_file=agent_storage_file)


# Vectordb - Fix URL format and add embedder here
api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
collection_name = "script_writer_knowledge"

vector_db = Qdrant(
    collection=collection_name,
    url=qdrant_url,
    api_key=api_key,
    embedder=GeminiEmbedder(id="text-embedding-004", 
                            dimensions=768, 
                            api_key=os.getenv("4DCNNGEMINI")),
)


# Configure Gemini models for chunking
chunking_model = Gemini(id="gemini-2.0-flash-lite", temperature=0.2, api_key=os.getenv("3DCNNGEMINI"))

# Knowledge Base - Configure with Gemini chunking
pdf_knowledge_base = PDFKnowledgeBase(
    path="/home/z4hid/Desktop/githubProjects/brainspark_agentic_workflow/knowledge/brainspark/brainspark.pdf",
    vector_db=vector_db,
    # reader=PDFReader(chunk=True, chunk_size=5000),
    chunking_strategy=AgenticChunking(model=chunking_model)
)


# Agents
script_writer = Agent(
    name="Script Writer",
    agent_id="script_writer",
    model=Gemini(id="gemini-2.0-flash", temperature=0.2, api_key=os.getenv("4DCNNGEMINI")),
    description="""
    The Script Writer agent specializes in creating scripts for various text-based and audio-visual media,
    including video content (e.g., promotional videos, explainer videos, webinars) and podcasts. 
    Its purpose is to adapt the core brand message, narrative elements, and relevant informational 
    content into formats suitable for these engaging mediums, thereby extending the reach and impact of BrainSpark Digital's content strategy.
""",
    instructions="""
    Adapt provided written content (such as blog posts, whitepapers, or case studies) into engaging and concise script formats suitable for the target medium (e.g., explainer videos, customer testimonial videos, educational webinars, podcast episodes).

    Ensure that the script faithfully maintains the core StoryBrand narrative structure: clearly identify the Hero (target viewer/listener), articulate their Problem, position BrainSpark Digital (or its client) as the Guide offering a Plan, and compellingly illustrate the path to Success.

    Break down complex information, technical details, or extensive narratives into digestible segments and clear language appropriate for audio or video consumption. Focus on clarity and ease of understanding.

    Write for the ear: employ conversational language, ensure a natural and engaging pacing, and utilize effective storytelling techniques to capture and retain audience attention.

    Include explicit cues within the script for visuals (e.g., specific graphics, B-roll footage suggestions, on-screen text overlays), sound effects, music, or presenter actions where appropriate to enhance the production value and message delivery.

    If scripting for SEO-driven video content (particularly for platforms like YouTube), naturally incorporate target keywords into the spoken script, as well as into prompts for video titles, descriptions, and tags.

    Develop comprehensive scripts for 'Educational Webinars and Online Workshops'. These scripts should focus on delivering practical skills, actionable insights, or information on emerging industry trends relevant to the target audience.
""",
    memory=memory,
    enable_user_memories=True,
    knowledge=pdf_knowledge_base,
    search_knowledge=True,
    add_references=True,
    enable_agentic_knowledge_filters=True,
    storage=agent_storage,
    add_history_to_messages=True,
    add_datetime_to_instructions=True,
    user_id="z4hid",
    debug_mode=True,
    markdown=True,
    role="script_writer",
    tools=[
        GoogleSearchTools(fixed_max_results=15),
        DuckDuckGoTools(fixed_max_results=10),
        TavilyTools(),
    ],
    show_tool_calls=True
)

if __name__ == "__main__":
    try:
        pdf_knowledge_base.load(recreate=False)
        # Comment out after first run
        
        script_writer.print_response("Write a 30 second video script for SEO guidelines from the SEO Specialist ", stream=True)
    except Exception as e:
        print(f"Error: {e}")
