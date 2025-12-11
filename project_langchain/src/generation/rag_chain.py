"""RAG chain builder for modular chain construction."""

from typing import Optional, Callable, Any
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever

from src.generation.prompt_templates import PromptTemplateManager, format_docs


class RAGChainBuilder:
    """Builder for creating modular RAG chains."""
    
    def __init__(
        self,
        retriever: BaseRetriever,
        llm: Any,
        prompt_template: Optional[PromptTemplate] = None,
        template_name: str = "rag",
        format_docs_func: Optional[Callable] = None
    ):
        """Initialize RAG chain builder.
        
        Args:
            retriever: LangChain retriever instance
            llm: LangChain LLM instance
            prompt_template: Optional custom prompt template
            template_name: Name of template to use if prompt_template not provided
            format_docs_func: Optional custom function to format documents
        """
        self.retriever = retriever
        self.llm = llm
        self.format_docs_func = format_docs_func or format_docs
        
        # Get prompt template
        if prompt_template:
            self.prompt = prompt_template
        else:
            self.prompt = PromptTemplateManager.get_template(template_name)
        
        # Build chain
        self.chain = self._build_chain()
    
    def _build_chain(self):
        """Build the RAG chain.
        
        Returns:
            Runnable chain
        """
        # Create parallel runnable for context and question
        runnable_parallel = RunnableParallel(
            {
                "context": self.retriever | self.format_docs_func,
                "question": RunnablePassthrough()
            }
        )
        
        # Build chain: prompt | llm | parser
        chain = self.prompt | self.llm | StrOutputParser()
        
        # Combine: parallel -> chain
        rag_chain = runnable_parallel.assign(answer=chain)
        
        return rag_chain
    
    def invoke(self, query: str) -> dict:
        """Invoke the RAG chain with a query.
        
        Args:
            query: User query string
            
        Returns:
            Dictionary with 'answer', 'context', and 'question' keys
        """
        return self.chain.invoke(query)
    
    def invoke_with_sources(self, query: str, has_good_info: bool = True) -> dict:
        """Invoke the RAG chain and include source information.
        
        Args:
            query: User query string
            has_good_info: Whether retrieved documents have good relevance scores
            
        Returns:
            Dictionary with 'answer', 'context', 'question', and 'sources' keys
        """
        # Get sources from retrieved documents
        retrieved_docs = self.retriever.invoke(query)
        sources = []
        for doc in retrieved_docs:
            course_code = doc.metadata.get('course_code', '')
            if course_code:
                sources.append(course_code)
        
        # If we don't have good information, modify the context to indicate this
        if not has_good_info or not retrieved_docs:
            # Create a modified context that tells the LLM we don't have good information
            context = "No relevant course information was found for this query. The retrieved documents are not sufficiently relevant to answer the question."
        else:
            # Format documents normally
            context = self.format_docs_func(retrieved_docs)
        
        # Format the prompt with context and question
        formatted_prompt = self.prompt.format(context=context, question=query)
        
        # Invoke LLM
        try:
            answer = self.llm.invoke(formatted_prompt)
            # Convert to string if needed
            if not isinstance(answer, str):
                answer = str(answer)
        except Exception as e:
            print(f"[WARNING] LLM invocation failed: {e}")
            answer = "I encountered an error while generating the answer."
        
        # If we don't have good info, ensure the answer reflects this
        if not has_good_info or not retrieved_docs:
            answer_lower = answer.lower() if isinstance(answer, str) else str(answer).lower()
            if "don't" not in answer_lower and "not available" not in answer_lower and "no information" not in answer_lower and "i don't know" not in answer_lower:
                answer = "I don't have sufficient information to answer this question based on the available course data."
        
        result = {
            'answer': str(answer),
            'context': context,
            'question': query,
            'sources': list(set(sources))  # Remove duplicates
        }
        
        return result
    
    @classmethod
    def create(
        cls,
        retriever: BaseRetriever,
        llm: Any,
        template_name: str = "rag",
        **kwargs
    ) -> "RAGChainBuilder":
        """Create a RAG chain builder.
        
        Args:
            retriever: LangChain retriever instance
            llm: LangChain LLM instance
            template_name: Name of prompt template to use
            **kwargs: Additional arguments
            
        Returns:
            RAGChainBuilder instance
        """
        return cls(
            retriever=retriever,
            llm=llm,
            template_name=template_name,
            **kwargs
        )


def create_rag_chain(
    retriever: BaseRetriever,
    llm: Any,
    template_name: str = "rag"
) -> RAGChainBuilder:
    """Convenience function to create a RAG chain.
    
    Args:
        retriever: LangChain retriever instance
        llm: LangChain LLM instance
        template_name: Name of prompt template to use
        
    Returns:
        RAGChainBuilder instance
    """
    return RAGChainBuilder.create(
        retriever=retriever,
        llm=llm,
        template_name=template_name
    )

