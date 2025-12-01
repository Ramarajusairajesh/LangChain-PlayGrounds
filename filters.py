from typing import List, Dict, Any, Optional, Union, Literal
from enum import Enum

from pydantic import BaseModel, Field
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                   SystemMessage)
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("presidio-analyzer not available. PII detection will be limited.")

class RedactionConfig(BaseModel):
    """Configuration for text redaction."""
    language: str = "en"
    entities: Optional[List[str]] = None
    redaction_operator: Dict[str, Any] = Field(
        default_factory=lambda: {"type": "replace", "new_value": "<REDACTED>"}
    )


class MessageType(str, Enum):
    HUMAN = "human"
    AI = "ai"
    SYSTEM = "system"
    OTHER = "other"


class RedactedMessage(BaseModel):
    """Model representing a redacted message with metadata."""
    content: str
    message_type: MessageType
    additional_kwargs: Dict[str, Any] = Field(default_factory=dict)
    is_redacted: bool = False


def redact_text(text: str, config: Optional[RedactionConfig] = None) -> tuple[str, bool]:
    """
    Redact sensitive information from text.
    
    Args:
        text: The text to redact
        config: Optional configuration for redaction
        
    Returns:
        tuple: (redacted_text, was_redacted) where was_redacted is True if any redaction occurred
    """
    if not text or not isinstance(text, str):
        return text, False
        
    if not SPACY_AVAILABLE:
        logger.warning("PII detection not available. Install presidio-analyzer for PII detection.")
        return text, False
        
    config = config or RedactionConfig()
    
    try:
        # Initialize the engines
        analyzer = AnalyzerEngine()
        anonymizer = AnonymizerEngine()
        
        # Analyze the text for PII
        results = analyzer.analyze(
            text=text,
            language=config.language,
            entities=config.entities
        )
        
        # If no PII found, return the original text
        if not results:
            return text, False
            
        # Anonymize the text
        redacted_text = anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators={
                "DEFAULT": config.redaction_operator
            }
        )
        
        return str(redacted_text), True
        
    except Exception as e:
        logger.error(f"Error during PII detection: {str(e)}")
        return text, False

def get_message_type(msg: BaseMessage) -> MessageType:
    """Determine the type of message."""
    if isinstance(msg, HumanMessage):
        return MessageType.HUMAN
    if isinstance(msg, AIMessage):
        return MessageType.AI
    if isinstance(msg, SystemMessage):
        return MessageType.SYSTEM
    return MessageType.OTHER


def create_typed_message(msg_type: MessageType, content: str, **kwargs) -> BaseMessage:
    """Create a typed message based on the message type."""
    if msg_type == MessageType.HUMAN:
        return HumanMessage(content=content, **kwargs)
    elif msg_type == MessageType.AI:
        return AIMessage(content=content, **kwargs)
    elif msg_type == MessageType.SYSTEM:
        return SystemMessage(content=content, **kwargs)
    else:
        return BaseMessage(content=content, **kwargs)


def redact_message(msg: BaseMessage, config: Optional[RedactionConfig] = None) -> tuple[BaseMessage, bool]:
    """
    Redact sensitive information from a message.
    
    Args:
        msg: The message to redact
        config: Optional configuration for redaction
        
    Returns:
        tuple: (redacted_message, was_redacted) where was_redacted is True if any redaction occurred
    """
    if not isinstance(msg.content, str):
        return msg, False
        
    redacted_content, was_redacted = redact_text(msg.content, config)
    
    # Create a new message of the same type with redacted content
    msg_type = get_message_type(msg)
    redacted_msg = create_typed_message(
        msg_type=msg_type,
        content=redacted_content,
        additional_kwargs=msg.additional_kwargs
    )
    
    return redacted_msg, was_redacted


def redact_messages(
    messages: List[BaseMessage], 
    config: Optional[RedactionConfig] = None
) -> tuple[List[BaseMessage], List[bool]]:
    """
    Redact sensitive information from a list of messages.
    
    Args:
        messages: List of messages to redact
        config: Optional configuration for redaction
        
    Returns:
        tuple: (list_of_redacted_messages, was_redacted_list) where was_redacted_list
               indicates which messages were redacted
    """
    redacted_messages = []
    was_redacted = []
    
    for msg in messages:
        redacted_msg, msg_redacted = redact_message(msg, config)
        redacted_messages.append(redacted_msg)
        was_redacted.append(msg_redacted)
    
    return redacted_messages, was_redacted



if __name__ == "__main__":
    # Example usage with custom configuration
    custom_config = RedactionConfig(
        entities=["PERSON", "EMAIL_ADDRESS"],  # Only redact names and emails
        redaction_operator={"type": "replace", "new_value": "[REDACTED]"}
    )
    
    original = HumanMessage(
        content="Hi, I am Ramesh and my email is ramesh@example.com and phone is 9876543210"
    )

    # Redact with default config
    redacted_default, was_redacted = redact_message(original)
    print("=== Default Redaction ===")
    print("Original:", original.content)
    print("Redacted:", redacted_default.content)
    print(f"Was redacted: {was_redacted}")
    
    # Redact with custom config
    redacted_custom, was_redacted = redact_message(original, custom_config)
    print("\n=== Custom Redaction ===")
    print("Original:", original.content)
    print("Redacted:", redacted_custom.content)
    print(f"Was redacted: {was_redacted}")
    
    # Example with multiple messages
    messages = [
        HumanMessage(content="My name is Alice and my number is 123-456-7890"),
        AIMessage(content="Hello Alice! How can I help you today?")
    ]
    
    print("\n=== Multiple Messages ===")
    redacted_msgs, redaction_status = redact_messages(messages, custom_config)
    for orig, redacted, was_redacted in zip(messages, redacted_msgs, redaction_status):
        print(f"\nOriginal: {orig.content}")
        print(f"Redacted: {redacted.content}")
        print(f"Was redacted: {was_redacted}")
