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
    """Format retrieved documents into a single context string, grouped by course.
    
    Groups documents by course code and formats them with sections.
    Each course shows metadata once, then all its sections.
    
    Args:
        docs: List of Document objects
        
    Returns:
        Formatted context string with courses grouped and sections organized
    """
    if not docs:
        return ""
    
    # Group documents by course code
    courses = {}
    for doc in docs:
        course_code = doc.metadata.get('course_code', 'Unknown')
        course_name = doc.metadata.get('course_name', '')
        course_type = doc.metadata.get('course_type', '')
        chunk_type = doc.metadata.get('chunk_type', 'unknown')
        section_name = doc.metadata.get('section_name_original', '')
        
        if course_code not in courses:
            courses[course_code] = {
                'name': course_name,
                'type': course_type,
                'sections': {}
            }
        
        # Extract section content (remove course context prefix)
        content = doc.page_content
        
        # Remove the "Section: ..." and "Course: ..." prefixes we added
        if chunk_type == 'section':
            # Extract just the section content
            lines = content.split('\n')
            content_lines = []
            found_content = False
            for line in lines:
                line_stripped = line.strip()
                # Skip header lines
                if line_stripped.startswith('Section:'):
                    continue
                elif line_stripped.startswith('Course:'):
                    continue
                elif line_stripped.startswith('Course Type:'):
                    continue
                elif not line_stripped:
                    # Skip empty lines before content starts
                    if not found_content:
                        continue
                else:
                    found_content = True
                    content_lines.append(line)
            content = '\n'.join(content_lines).strip()
            
            # Map section names to standard names
            section_key = section_name.lower()
            if 'prerequisite' in section_key or 'prerequisit' in section_key:
                section_key = 'prerequisites'
            elif 'learning outcome' in section_key:
                section_key = 'learning outcomes'
            elif 'examination' in section_key or 'exam' in section_key:
                section_key = 'examination'
            elif 'organisation' in section_key or 'organization' in section_key:
                section_key = 'organisation'
            elif 'literature' in section_key:
                section_key = 'literature'
            elif 'overview' in section_key:
                section_key = 'overview'
            elif 'content' in section_key:
                section_key = 'content'
            else:
                section_key = section_name.lower()
            
            if section_key not in courses[course_code]['sections']:
                courses[course_code]['sections'][section_key] = []
            # Only add if content is not already in the list (deduplicate)
            if content not in courses[course_code]['sections'][section_key]:
                courses[course_code]['sections'][section_key].append(content)
        elif chunk_type == 'metadata':
            # Store metadata separately
            courses[course_code]['metadata'] = content
    
    # Format courses
    formatted_parts = []
    for course_code, course_data in courses.items():
        course_name = course_data['name']
        course_type = course_data['type']
        
        # Course header
        formatted_parts.append(f"### Course: {course_code} – {course_name}")
        if course_type:
            formatted_parts.append(f"Type: {course_type}")
        formatted_parts.append("")
        
        # Add sections in order
        section_order = ['overview', 'prerequisites', 'learning outcomes', 'content', 
                        'examination', 'organisation', 'literature']
        
        for section_key in section_order:
            if section_key in course_data['sections']:
                # Capitalize section name
                section_display = section_key.replace('_', ' ').title()
                formatted_parts.append(f"#### {section_display}")
                # Join multiple chunks for same section (deduplicate first)
                section_contents = course_data['sections'][section_key]
                # Remove exact duplicates
                unique_contents = []
                seen = set()
                for content in section_contents:
                    content_normalized = content.strip()
                    if content_normalized and content_normalized not in seen:
                        seen.add(content_normalized)
                        unique_contents.append(content)
                section_content = '\n\n'.join(unique_contents) if unique_contents else section_contents[0]
                formatted_parts.append(section_content)
                formatted_parts.append("")
        
        # Add any remaining sections not in the standard order
        for section_key, section_contents in course_data['sections'].items():
            if section_key not in section_order:
                section_display = section_key.replace('_', ' ').title()
                formatted_parts.append(f"#### {section_display}")
                # Remove exact duplicates
                unique_contents = []
                seen = set()
                for content in section_contents:
                    content_normalized = content.strip()
                    if content_normalized and content_normalized not in seen:
                        seen.add(content_normalized)
                        unique_contents.append(content)
                section_content = '\n\n'.join(unique_contents) if unique_contents else section_contents[0]
                formatted_parts.append(section_content)
                formatted_parts.append("")
    
    return '\n'.join(formatted_parts)

