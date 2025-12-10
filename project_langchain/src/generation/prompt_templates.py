"""Prompt template management for RAG chains."""

from typing import Dict, Optional
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


class PromptTemplateManager:
    """Manager for prompt templates used in RAG chains."""
    
    # Template definitions
    RAG_TEMPLATE = """You are a helpful assistant that answers questions about university courses. Use the following course information to answer the question. If the information is not available in the context, say so.

Course Information:
{context}

Question: {question}

Answer:"""
    
    RAG_TEMPLATE_BRIEF = """Answer the question based only on the context below. Be very brief and specific (1-3 sentences).

Context:
{context}

Question: {question}

Answer:"""
    
    QA_TEMPLATE = """Based on the following course information, answer the question. If you cannot answer from the provided information, say "I don't have enough information to answer this question."

Course Information:
{context}

Question: {question}

Answer:"""
    
    @staticmethod
    def get_rag_template(template_type: str = "standard") -> PromptTemplate:
        """Get RAG prompt template.
        
        Args:
            template_type: Type of template ("standard" or "brief")
            
        Returns:
            PromptTemplate instance
        """
        if template_type == "brief":
            return PromptTemplate.from_template(PromptTemplateManager.RAG_TEMPLATE_BRIEF)
        else:
            return PromptTemplate.from_template(PromptTemplateManager.RAG_TEMPLATE)
    
    @staticmethod
    def get_qa_template() -> PromptTemplate:
        """Get QA prompt template.
        
        Returns:
            PromptTemplate instance
        """
        return PromptTemplate.from_template(PromptTemplateManager.QA_TEMPLATE)
    
    @staticmethod
    def get_chat_template(model_name: Optional[str] = None) -> ChatPromptTemplate:
        """Get chat-style prompt template (for chat models).
        
        Args:
            model_name: Optional model name to determine format
            
        Returns:
            ChatPromptTemplate instance
        """
        # Default to Llama 3.2 format
        if model_name and "llama" in model_name.lower():
            return ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant that answers questions about university courses. Use the following course information to answer the question. If the information is not available in the context, say so."),
                ("user", "Course Information:\n{context}\n\nQuestion: {question}")
            ])
        else:
            # Generic chat format
            return ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant that answers questions about university courses."),
                ("user", "Course Information:\n{context}\n\nQuestion: {question}")
            ])
    
    @staticmethod
    def create_custom_template(template_string: str) -> PromptTemplate:
        """Create a custom prompt template from a string.
        
        Args:
            template_string: Template string with {context} and {question} placeholders
            
        Returns:
            PromptTemplate instance
        """
        return PromptTemplate.from_template(template_string)
    
    @staticmethod
    def get_template(template_name: str, **kwargs) -> PromptTemplate:
        """Get a template by name.
        
        Args:
            template_name: Name of template ("rag", "rag_brief", "qa", "chat")
            **kwargs: Additional arguments for template creation
            
        Returns:
            PromptTemplate or ChatPromptTemplate instance
        """
        if template_name == "rag":
            return PromptTemplateManager.get_rag_template("standard")
        elif template_name == "rag_brief":
            return PromptTemplateManager.get_rag_template("brief")
        elif template_name == "qa":
            return PromptTemplateManager.get_qa_template()
        elif template_name == "chat":
            return PromptTemplateManager.get_chat_template(kwargs.get("model_name"))
        else:
            raise ValueError(f"Unknown template: {template_name}")


def format_docs(docs) -> str:
    """Format retrieved documents into a single context string.
    
    Args:
        docs: List of Document objects
        
    Returns:
        Formatted context string
    """
    return "\n\n".join(doc.page_content for doc in docs)

