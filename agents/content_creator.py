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

memory_db = SqliteMemoryDb(table_name="content_creator_memory", 
                           db_file=memory_storage_file)

# Create Memory object that wraps the database
memory = Memory(db=memory_db)

# Storage
agent_storage = SqliteStorage(table_name="content_creator_agent", db_file=agent_storage_file)


# Vectordb - Fix URL format and add embedder here
api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
collection_name = "content_creator_knowledge"

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
    path=knowledge_dir.joinpath("brainspark.pdf"),
    vector_db=vector_db,
    # reader=PDFReader(chunk=True, chunk_size=5000),
    chunking_strategy=AgenticChunking(model=chunking_model)
)


# Agents
content_creator = Agent(
    name="Content Creator",
    agent_id="content_creator",
    model=Gemini(id="gemini-2.0-flash", temperature=0.2, api_key=os.getenv("3DCNNGEMINI")),
    description="""
    The Content Creator agent is responsible for generating various forms of high-quality written content. This includes, 
    but is not limited to, blog posts, website copy, articles, whitepapers, and case studies. Its primary purpose is to translate strategic 
    inputs from the Brandscript Architect and SEO Specialist into compelling, informative, and optimized content that engages the 
    target audience and supports BrainSpark Digital's marketing objectives.
""",
    instructions="""
    Upon receiving an MVC brief or content request, thoroughly analyze the provided Brandscript elements (Hero, Problem, Solution, etc.) and the target SEO keywords and user intent.

    Draft content that compellingly embodies the StoryBrand narrative. Consistently position the intended reader as the Hero of the story, clearly address their Problems (external, internal, philosophical), introduce BrainSpark Digital (or its client) as the trusted Guide offering a viable Plan, and vividly articulate the path to Success while also highlighting the consequences of Failure.

    Naturally and strategically integrate primary and secondary keywords throughout the content. Adhere to on-page SEO best practices regarding keyword density, placement in headings, inclusion in the first 100 words, and semantic relevance, as per the guidelines provided by the SEO Specialist.

    Ensure all generated content is structured for optimal readability and user engagement. Employ techniques such as short paragraphs, bullet points, numbered lists, and clear subheadings. The content must provide genuine value and comprehensively address the core questions or needs of the target audience.

    Craft compelling headlines and introductions that are benefit-driven, incorporate target keywords effectively, and directly address the identified user intent for the given topic.

    Include clear and persuasive Calls to Action (CTAs), both Direct (e.g., 'Request a Demo') and Transitional (e.g., 'Download our e-book on AI strategy'), as defined by the Brandscript Architect and relevant to the content's purpose.

    If tasked with creating 'Pillar Content' and associated 'Topic Clusters', ensure that appropriate internal linking strategies are planned and placeholders or instructions for these links are included in the draft.

    When developing 'How-to' guides or tutorials, focus on providing clear, actionable advice and step-by-step instructions that readers can easily follow.

    If generating 'Case Studies', structure them as compelling narratives. Highlight the client's journey: their initial Challenge (Problem), how BrainSpark Digital acted as their Guide and implemented a Plan, and the remarkable, quantifiable Results and Transformation (Success) they achieved.

    Leverage the Retrieval Augmented Generation (RAG) capability to access and incorporate specific facts, statistics, examples, or company-approved information from the pre-processed 'Data Sources' to enhance the factual accuracy, depth, and credibility of the content.
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
    role="content_creator",
    tools=[
        GoogleSearchTools(fixed_max_results=15),
        DuckDuckGoTools(fixed_max_results=10),
        TavilyTools(),
        WikipediaTools(),
        FirecrawlTools(),
    ],
    show_tool_calls=True
)

if __name__ == "__main__":
    try:
        pdf_knowledge_base.load(recreate=False)
        # Comment out after first run
        
        content_creator.print_response("Write a 1500-word blog post on the future of AI in web development, targeting the keyword 'AI-driven web design trends", stream=True)
    except Exception as e:
        print(f"Error: {e}")
