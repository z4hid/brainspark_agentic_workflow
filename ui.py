import chainlit as cl
from agents.storybrand import brandscript_architect
from agents.seo import seo_specialist
from agents.product_manager import product_manager
from agents.growth_hacker import growth_hacker
from agents.social_media_manager import social_media_manager
from agents.script_writer import script_writer
from agents.content_creator import content_creator
import asyncio
from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store agent instances with enhanced metadata
AGENTS = {
    "BrandScript Architect": {
        "agent": brandscript_architect,
        "icon": "🎯",
        "color": "#8B5CF6",
        "expertise": "Brand Strategy & StoryBrand Framework"
    },
    "SEO Specialist": {
        "agent": seo_specialist,
        "icon": "🔍", 
        "color": "#10B981",
        "expertise": "Search Engine Optimization & Lean SEO"
    },
    "Product Manager": {
        "agent": product_manager,
        "icon": "📊",
        "color": "#3B82F6", 
        "expertise": "Product Strategy & Market Analysis"
    },
    "Growth Hacker": {
        "agent": growth_hacker,
        "icon": "📈",
        "color": "#F59E0B",
        "expertise": "Growth Strategy & AARRR Funnel"
    },
    "Social Media Manager": {
        "agent": social_media_manager,
        "icon": "📱",
        "color": "#EF4444",
        "expertise": "Social Media Strategy & Engagement"
    },
    "Script Writer": {
        "agent": script_writer,
        "icon": "🎬",
        "color": "#8B5CF6",
        "expertise": "Video Scripts & Content Creation"
    },
    "Content Creator": {
        "agent": content_creator,
        "icon": "✍️",
        "color": "#06B6D4",
        "expertise": "Blog Posts & Marketing Content"
    }
}

@cl.set_chat_profiles
async def chat_profile():
    """Set up different agent profiles for the UI with enhanced styling."""
    profiles = []
    
    for name, agent_info in AGENTS.items():
        profiles.append(
            cl.ChatProfile(
                name=name,
                markdown_description=f"**{agent_info['expertise']}**\n\n{agent_info['agent'].description[:200]}..." if len(agent_info['agent'].description) > 200 else agent_info['agent'].description,
                icon=agent_info['icon']
            )
        )
    
    return profiles

@cl.set_starters
async def set_starters():
    """Set up contextual starter prompts based on selected agent."""
    chat_profile = cl.user_session.get("chat_profile", "BrandScript Architect")
    
    # Define starters for each agent
    agent_starters = {
        "BrandScript Architect": [
            cl.Starter(
                label="🎯 Create Master BrandScript",
                message="Help me create a comprehensive StoryBrand (SB7) Master BrandScript for my business",
                icon="🎯"
            ),
            cl.Starter(
                label="👥 Define Customer Character",
                message="Help me identify and define my ideal customer character",
                icon="👥"
            ),
            cl.Starter(
                label="⚡ Craft Problem Statement",
                message="Help me articulate the problems my customers face (External, Internal, Philosophical)",
                icon="⚡"
            ),
            cl.Starter(
                label="🏆 Success Transformation",
                message="Help me paint a vivid picture of customer success and transformation",
                icon="🏆"
            )
        ],
        "SEO Specialist": [
            cl.Starter(
                label="🔍 SEO Website Audit",
                message="Analyze my website's SEO performance and provide optimization recommendations",
                icon="🔍"
            ),
            cl.Starter(
                label="🎯 Keyword Research",
                message="Conduct comprehensive keyword research for my business niche",
                icon="🎯"
            ),
            cl.Starter(
                label="📈 Lean SEO Strategy",
                message="Develop a Lean SEO strategy with quick wins and long-term goals",
                icon="📈"
            ),
            cl.Starter(
                label="🔧 Technical SEO Issues",
                message="Identify and prioritize technical SEO issues on my website",
                icon="🔧"
            )
        ],
        "Product Manager": [
            cl.Starter(
                label="📊 Market Analysis",
                message="Analyze market trends and opportunities for my digital services",
                icon="📊"
            ),
            cl.Starter(
                label="🎯 Service Strategy",
                message="Help me develop and refine my service offerings strategy",
                icon="🎯"
            ),
            cl.Starter(
                label="💰 Pricing Strategy",
                message="Create A/B testing strategies for service packages and pricing",
                icon="💰"
            ),
            cl.Starter(
                label="🚀 Product Roadmap",
                message="Develop a strategic roadmap for scaling our service delivery",
                icon="🚀"
            )
        ],
        "Growth Hacker": [
            cl.Starter(
                label="📈 AARRR Funnel Analysis",
                message="Analyze my customer acquisition funnel and identify growth opportunities",
                icon="📈"
            ),
            cl.Starter(
                label="🚀 Growth Experiments",
                message="Design growth hacking experiments to increase user acquisition",
                icon="🚀"
            ),
            cl.Starter(
                label="💡 Retention Strategy",
                message="Develop strategies to improve customer retention and reduce churn",
                icon="💡"
            ),
            cl.Starter(
                label="🎯 Conversion Optimization",
                message="Optimize conversion rates across different customer touchpoints",
                icon="🎯"
            )
        ],
        "Social Media Manager": [
            cl.Starter(
                label="📱 Social Media Strategy",
                message="Create a comprehensive social media strategy for my brand",
                icon="📱"
            ),
            cl.Starter(
                label="📅 Content Calendar",
                message="Develop a strategic content calendar for social media platforms",
                icon="📅"
            ),
            cl.Starter(
                label="🎯 Audience Engagement",
                message="Improve audience engagement and community building strategies",
                icon="🎯"
            ),
            cl.Starter(
                label="📊 Performance Analytics",
                message="Analyze social media performance and optimize content strategy",
                icon="📊"
            )
        ],
        "Script Writer": [
            cl.Starter(
                label="🎬 Video Script Creation",
                message="Write engaging video scripts for marketing campaigns",
                icon="🎬"
            ),
            cl.Starter(
                label="📺 Explainer Videos",
                message="Create scripts for explainer videos about our services",
                icon="📺"
            ),
            cl.Starter(
                label="🎙️ Podcast Scripts",
                message="Develop podcast episode scripts and interview questions",
                icon="🎙️"
            ),
            cl.Starter(
                label="📢 Ad Copy Scripts",
                message="Write compelling scripts for video advertisements",
                icon="📢"
            )
        ],
        "Content Creator": [
            cl.Starter(
                label="✍️ Blog Content Strategy",
                message="Develop a comprehensive blog content strategy",
                icon="✍️"
            ),
            cl.Starter(
                label="📖 Article Writing",
                message="Create engaging articles for thought leadership",
                icon="📖"
            ),
            cl.Starter(
                label="📧 Email Campaigns",
                message="Design email marketing campaigns and newsletter content",
                icon="📧"
            ),
            cl.Starter(
                label="🎯 Lead Magnets",
                message="Create valuable lead magnets and downloadable content",
                icon="🎯"
            )
        ]
    }
    
    return agent_starters.get(chat_profile, agent_starters["BrandScript Architect"])

@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session with the selected agent."""
    chat_profile = cl.user_session.get("chat_profile", "BrandScript Architect")
    
    # Get the selected agent info
    agent_info = AGENTS.get(chat_profile)
    if not agent_info:
        agent_info = AGENTS["BrandScript Architect"]  # Default fallback
    
    agent = agent_info["agent"]
    
    # Store agent and info in session
    cl.user_session.set("agent", agent)
    cl.user_session.set("agent_info", agent_info)
    cl.user_session.set("chat_profile", chat_profile)
    
    # Create an enhanced welcome message with agent-specific styling
    welcome_msg = f"""
# {agent_info['icon']} Welcome to BrainSpark Digital! 

## You're chatting with: **{chat_profile}** {agent_info['icon']}

### **Expertise:** {agent_info['expertise']}

---

**What I can help you with:**
{agent.description[:300]}...

---

💡 **Tip:** Use the starter prompts below to get the conversation rolling, or ask me anything related to my expertise!

🔄 **Switch Agents:** You can change to a different specialist anytime using the profile dropdown.
    """
    
    # Send welcome message with enhanced styling
    welcome_element = cl.Message(
        content=welcome_msg,
        author=chat_profile,
    )
    await welcome_element.send()

@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages and stream agent responses with enhanced UX."""
    
    agent = cl.user_session.get("agent")
    agent_info = cl.user_session.get("agent_info")
    chat_profile = cl.user_session.get("chat_profile", "BrandScript Architect")
    
    if not agent or not agent_info:
        await cl.Message(
            content="❌ Agent not initialized. Please refresh the page and select an agent.",
            author="System"
        ).send()
        return
    
    # Create main response message with agent branding
    response_msg = cl.Message(
        content="",
        author=f"{agent_info['icon']} {chat_profile}"
    )
    
    # Create enhanced status step with agent-specific branding
    step_name = f"{agent_info['icon']} {chat_profile} Processing"
    async with cl.Step(name=step_name, type="llm") as status_step:
        status_step.input = message.content
        
        try:
            # Agent-specific status messages
            status_messages = {
                "BrandScript Architect": [
                    "🎯 **Activating StoryBrand framework...**",
                    "📚 **Searching brand strategy knowledge...**",
                    "🧠 **Analyzing customer journey...**",
                    "✨ **Crafting compelling narrative...**"
                ],
                "SEO Specialist": [
                    "🔍 **Initializing SEO analysis...**",
                    "📊 **Gathering Lean SEO insights...**",
                    "🎯 **Analyzing keyword opportunities...**",
                    "🚀 **Optimizing search strategy...**"
                ],
                "Product Manager": [
                    "📈 **Analyzing market trends...**",
                    "🎯 **Evaluating product strategy...**",
                    "💡 **Researching competitive landscape...**",
                    "🚀 **Formulating recommendations...**"
                ],
                "Growth Hacker": [
                    "📊 **Analyzing AARRR funnel...**",
                    "🚀 **Identifying growth opportunities...**",
                    "🎯 **Designing experiments...**",
                    "📈 **Optimizing conversion paths...**"
                ],
                "Social Media Manager": [
                    "📱 **Analyzing social trends...**",
                    "🎯 **Researching audience insights...**",
                    "📅 **Planning content strategy...**",
                    "🚀 **Optimizing engagement...**"
                ],
                "Script Writer": [
                    "🎬 **Activating creative engine...**",
                    "📝 **Structuring narrative flow...**",
                    "🎭 **Crafting compelling dialogue...**",
                    "✨ **Polishing script elements...**"
                ],
                "Content Creator": [
                    "✍️ **Initializing content strategy...**",
                    "📚 **Researching topic insights...**",
                    "🎯 **Structuring content flow...**",
                    "🚀 **Optimizing for engagement...**"
                ]
            }
            
            # Get agent-specific status messages or use default
            agent_status = status_messages.get(chat_profile, status_messages["BrandScript Architect"])
            
            # Display status updates with timing
            for i, status_msg in enumerate(agent_status):
                await status_step.stream_token(f"{status_msg}\n")
                await asyncio.sleep(0.7 + (i * 0.2))  # Progressive timing
            
            # Execute agent with enhanced error handling
            response = agent.run(message.content)
            
            await status_step.stream_token("✅ **Analysis complete!**\n")
            
            # Handle the response content
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            # Enhanced streaming with variable timing for better UX
            words = content.split()
            current_chunk = ""
            
            for i, word in enumerate(words):
                current_chunk += word + " "
                
                # Stream in natural word groups (5-8 words)
                if i % 6 == 0 and i > 0:
                    await response_msg.stream_token(current_chunk)
                    current_chunk = ""
                    await asyncio.sleep(0.03)  # Natural reading pace
            
            # Stream any remaining content
            if current_chunk:
                await response_msg.stream_token(current_chunk)
                    
        except Exception as e:
            logger.error(f"Error during agent execution: {e}")
            await status_step.stream_token(f"❌ **Error: {str(e)}**\n")
            
            # Enhanced error message with recovery suggestions
            error_msg = f"""
❌ **Something went wrong!** 

**Error Details:** {str(e)[:100]}...

**Try this:**
- Refresh the page and try again
- Simplify your question
- Switch to a different agent
- Check your internet connection

**Need help?** The issue has been logged for our team to review.
            """
            
            await cl.Message(
                content=error_msg,
                author="System"
            ).send()
            return
        
        # Finalize the status step with success message
        status_step.output = f"✅ Successfully processed query with {chat_profile}"
    
    # Send the final response
    await response_msg.send()
    
    # Add helpful follow-up suggestions
    follow_up_msg = f"""
---
💬 **Continue the conversation:**
- Ask for clarification on any point
- Request specific examples or case studies  
- Explore related topics within my expertise
- Switch to another specialist for different perspectives

{agent_info['icon']} **{chat_profile}** is ready for your next question!
    """
    
    await cl.Message(
        content=follow_up_msg,
        author="Assistant"
    ).send()

@cl.on_chat_end
async def on_chat_end():
    """Clean up when chat ends."""
    logger.info("Chat session ended")
    # Chainlit handles session cleanup automatically
    # No need to manually clear the session

@cl.on_stop
async def on_stop():
    """Handle when user stops the agent."""
    agent_info = cl.user_session.get("agent_info", {"icon": "🤖"})
    await cl.Message(
        content=f"⏹️ **Task stopped by user.** {agent_info['icon']} Ready for your next question!",
        author="Assistant"
    ).send()

@cl.on_settings_update
async def setup_agent_settings(settings):
    """Handle settings updates for enhanced user control."""
    # Store settings in user session
    cl.user_session.set("settings", settings)
    
    # Apply settings to current agent if needed
    agent = cl.user_session.get("agent")
    if agent and settings:
        # You can modify agent behavior based on settings here
        logger.info(f"Settings updated: {settings}")

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    """Optional: Add authentication if needed in production."""
    # For demo purposes, accept any credentials
    # In production, implement proper authentication
    return cl.User(
        identifier=username, 
        metadata={"role": "user", "provider": "credentials"}
    )

if __name__ == "__main__":
    # This allows running the UI directly
    import os
    import subprocess
    
    # Run chainlit
    subprocess.run(["chainlit", "run", __file__, "-w"]) 