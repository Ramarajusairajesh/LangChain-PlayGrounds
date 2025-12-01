import asyncio
import logging
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import filters
except ImportError:
    logger.warning("filters.py not found. Some features may be limited.")
    filters = None


class NewsConfig(BaseModel):
    model_name: str = "gemini-pro"  # Using the stable model
    temperature: float = 0.7
    system_message: str = "You are a helpful news assistant."
    google_api_key: str = ""  # Add this line to store the API key


class NewsContext(BaseModel):
    user_role: str = "General"


class NewsAgent:
    
    def __init__(self, config: Optional[NewsConfig] = None):
        self.config = config or NewsConfig()
        self.llm = self._initialize_llm()
        self.chain = self._create_chain()
    
    def _initialize_llm(self):
        """Initialize the language model."""
        if not self.config.google_api_key:
            logger.error("Google API key is not set. Please set the GOOGLE_API_KEY environment variable.")
            raise ValueError("Google API key is required. Please set the GOOGLE_API_KEY environment variable.")
            
        try:
            return ChatGoogleGenerativeAI(
                model=self.config.model_name,
                temperature=self.config.temperature,
                google_api_key=self.config.google_api_key
            )
        except Exception as e:
            logger.error(f"Failed to initialize language model: {str(e)}")
            raise RuntimeError(
                "Failed to start the language model. "
                "Please check your API key and internet connection. If neither are the wrong, create a new api key and try again."
            ) from e
    
    def _get_system_prompt(self, user_role: str) -> str:
        base_prompt = "You are a news reporter"
        role_prompts = {
            "Sports": (
                f"{base_prompt} specializing in sports news. "
                "Provide detailed sports coverage, scores, and player updates."
            ),
            "Financial": (
                f"{base_prompt} specializing in financial news. "
                "Provide market analysis, stock updates, and economic trends."
            ),
            "Politics": (
                f"{base_prompt} specializing in political news. "
                "Provide political analysis, policy updates, and election coverage."
            ),
            "Tech": (
                f"{base_prompt} specializing in technology news. "
                "Provide updates on the latest tech releases, startups, and innovations."
            )
        }
        
        return role_prompts.get(user_role, 
            f"{base_prompt} covering general news. "
            "Provide balanced coverage of current events and trending topics."
        )
    
    def _create_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("Gemini", "{system_prompt}"),
            ("human", "{question}")
        ])
        return prompt | self.llm
    async def ask_news(self, topic: str, user_question: str) -> str:
        try:
            system_prompt = self._get_system_prompt(topic)
            response = await self.chain.ainvoke({
                "system_prompt": system_prompt,
                "question": user_question
            })
            
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            logger.error(f"Error in ask_news: {str(e)}", exc_info=True)
            return f"I'm sorry, I encountered an error: {str(e)}"


news_agent: Optional[NewsAgent] = None


async def ask_news(topic: str, user_question: str) -> str:
    if news_agent is None:
        raise RuntimeError("NewsAgent has not been initialized.")
    return await news_agent.ask_news(topic, user_question)


async def interactive_news_loop():
    print("News Agent")
    print("Type 'exit' as topic to quit.\n")

    while True:
        try:
            topic = input(
                "Topic (Sports/Financial/Politics/Tech/General or 'help/exit'): "
            ).strip()
            
            if topic.lower().strip() == "help":
                await example_single_call()
                continue
                
            if topic.lower() == "exit":
                print("Goodbye!")
                break

            user_question = input("Your question: ").strip()
            if not user_question:
                print("Please enter a question.\n")
                continue

            try:
                if filters and hasattr(filters, 'redact_message'):
                    try:
                        redacted_msg, _ = filters.redact_message(HumanMessage(content=user_question))
                        if hasattr(redacted_msg, 'content'):
                            print(f"Processed question: {redacted_msg.content}")
                    except Exception as e:
                        logger.warning(f"Error in redaction: {str(e)}")
                        print(f"Original question: {user_question}")
                answer = await ask_news(topic=topic, user_question=user_question)
                print(f"\n[{topic.upper()} NEWS]")
                print(answer)
                print("\n" + "-" * 50 + "\n")
            except Exception as e:
                print(f"Error while processing your request: {str(e)}\n")
                logger.exception("Error in interactive loop")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting news agent. Goodbye!")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}\n")
            logger.exception("Unexpected error in interactive_news_loop")


def example_single_call():
    try:
        topic = "Sports"
        question = "What's the score of India in the recent T20?"
        
        print(f"[EXAMPLE] Asking about {topic}: {question}")
        
        # Using asyncio.run to run the async function in a synchronous context
        answer = asyncio.run(ask_news(topic=topic, user_question=question))
        
        print(f"[RESPONSE] {answer}")
        
    except Exception as e:
        print(f"Example error: {str(e)}")
        logger.exception("Error in example_single_call")


async def main():
    try:
        import os
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            print("Error: GOOGLE_API_KEY environment variable is not set.")
            print("Please set it by running:")
            print("export GOOGLE_API_KEY='your-api-key-here'")
            return
        global news_agent
        config = NewsConfig(google_api_key=google_api_key)
        news_agent = NewsAgent(config=config)
        await interactive_news_loop()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nA critical error occurred: {str(e)}")
        logger.critical("Critical error in main", exc_info=True)
    finally:
        print("\nThank you for using the News Agent!")

if __name__ == "__main__":
    asyncio.run(main())
