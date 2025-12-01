import os
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field, field_validator, SecretStr
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WeatherConfig(BaseModel):
    """Configuration for the weather chat application."""
    model_name: str = Field(
        default="gemini-2.5-flash",
        description="Name of the Gemini model to use"
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Controls randomness in the model's responses (0.0 to 1.0)"
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        description="Maximum number of retries for API calls"
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose output for debugging"
    )


class MessageRole(str, Enum):
    """Roles for chat messages."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """A chat message with role and content."""
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_langchain(cls, message: BaseMessage) -> 'ChatMessage':
        """Convert a LangChain message to a ChatMessage."""
        if isinstance(message, SystemMessage):
            role = MessageRole.SYSTEM
        elif isinstance(message, HumanMessage):
            role = MessageRole.USER
        elif isinstance(message, AIMessage):
            role = MessageRole.ASSISTANT
        else:
            role = MessageRole.USER  # Default to user for unknown types
            
        return cls(
            role=role,
            content=message.content if hasattr(message, 'content') else str(message)
        )


class WeatherAssistant:
    """A weather assistant that uses Gemini for weather-related queries."""
    
    def __init__(self, config: Optional[WeatherConfig] = None):
        """Initialize the weather assistant with configuration."""
        self.config = config or WeatherConfig()
        self._api_key: Optional[SecretStr] = self._get_api_key()
        self.llm = self._initialize_llm()
        self.conversation_history: List[ChatMessage] = self._get_initial_system_message()
    
    def _get_api_key(self) -> SecretStr:
        """Get the API key from environment or configuration."""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
            raise ValueError(
                "Please set the GOOGLE_API_KEY environment variable or provide it in the configuration."
            )
        return SecretStr(api_key)
    
    def _initialize_llm(self) -> ChatGoogleGenerativeAI:
        """Initialize the language model with retry logic."""
        try:
            return ChatGoogleGenerativeAI(
                model=self.config.model_name,
                temperature=self.config.temperature,
                api_key=self._api_key.get_secret_value() if self._api_key else None,
                max_retries=self.config.max_retries,
            )
        except Exception as e:
            logger.error(f"Failed to initialize language model: {str(e)}")
            raise RuntimeError(
                "Failed to initialize the language model. Please check your API key and internet connection."
            ) from e
    
    def _get_initial_system_message(self) -> List[ChatMessage]:
        """Get the initial system message for the conversation."""
        system_content = (
            "You are a helpful weather assistant. "
            "You DO NOT have access to live weather APIs or real-time data. "
            "Base your answers on general climate patterns and historical norms, "
            "and clearly say that you are estimating rather than giving live data. "
            f"The current date is {datetime.now().strftime('%A, %B %d, %Y')}. "
            f"The current time is {datetime.now().strftime('%I:%M %p')} UTC."
        )
        return [ChatMessage(role=MessageRole.SYSTEM, content=system_content)]
    
    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation history."""
        if not content or not content.strip():
            raise ValueError("Message content cannot be empty")
        self.conversation_history.append(
            ChatMessage(role=MessageRole.USER, content=content.strip())
        )
    
    def get_ai_response(self) -> str:
        """Get a response from the AI based on the conversation history."""
        try:
            # Convert ChatMessage objects back to LangChain message format
            langchain_messages = []
            for msg in self.conversation_history:
                if msg.role == MessageRole.SYSTEM:
                    langchain_messages.append(SystemMessage(content=msg.content))
                elif msg.role == MessageRole.USER:
                    langchain_messages.append(HumanMessage(content=msg.content))
                else:
                    langchain_messages.append(AIMessage(content=msg.content))
            
            # Get response from the language model
            response = self.llm.invoke(langchain_messages)
            
            # Add the AI's response to the conversation history
            ai_message = ChatMessage(
                role=MessageRole.ASSISTANT,
                content=response.content
            )
            self.conversation_history.append(ai_message)
            
            return response.content
            
        except Exception as e:
            logger.error(f"Error getting AI response: {str(e)}")
            error_msg = "I'm sorry, I encountered an error while processing your request. Please try again later."
            self.conversation_history.append(
                ChatMessage(role=MessageRole.ASSISTANT, content=error_msg)
            )
            return error_msg

def display_welcome_message() -> None:
    """Display welcome message and usage instructions."""
    print("\n" + "=" * 60)
    print("  Weather Assistant".center(60))
    print("=" * 60)
    print("\nI can provide information about typical weather patterns and climate norms.")
    print("Note: I don't have access to live weather data.")
    print("\nExamples:")
    print("  • What's the typical weather in Hyderabad in November?")
    print("  • Is it usually rainy in Bangalore in July?")
    print("  • What should I pack for a trip to Delhi in December?")
    print("\nType 'quit' or 'exit' to end the conversation.\n")


def run_interactive_chat() -> None:
    """Run the interactive chat interface."""
    try:
        # Initialize the weather assistant
        assistant = WeatherAssistant()
        display_welcome_message()
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                # Handle exit commands
                if user_input.lower() in {"quit", "exit"}:
                    print("\nThank you for using the Weather Assistant. Goodbye!")
                    break
                    
                if not user_input:
                    continue
                
                # Add user message and get response
                assistant.add_user_message(user_input)
                response = assistant.get_ai_response()
                
                # Display the response with some formatting
                print("\n" + "-" * 60)
                print(f"Assistant: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\nTo exit, type 'quit' or 'exit' and press Enter.")
                print("To continue, just type your next question.\n")
                continue
                
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                print("\nSorry, I encountered an unexpected error. Please try again.")
                continue
                
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}", exc_info=True)
        print("\nA critical error occurred. Please check the logs for more information.")
        print("Error:", str(e))


def main() -> None:
    """Main entry point for the weather assistant."""
    try:
        run_interactive_chat()
    except KeyboardInterrupt:
        print("\n\nExiting Weather Assistant. Goodbye!")
    except Exception as e:
        logger.critical(f"Unhandled exception: {str(e)}", exc_info=True)
        print("\nAn unexpected error occurred. Please check the logs for details.")
        raise


if __name__ == "__main__":
    main()
