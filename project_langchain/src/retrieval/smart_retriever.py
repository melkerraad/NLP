"""Smart retriever with dynamic K and metadata filtering."""

import re
from typing import List, Dict, Optional, Tuple
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_chroma import Chroma


class QueryAnalyzer:
    """Analyze queries to extract course codes and determine query type."""
    
    # Pattern to match course codes (e.g., DAT450, TMA947)
    COURSE_CODE_PATTERN = re.compile(r'\b([A-Z]{3}\d{3})\b', re.IGNORECASE)
    
    # Keywords that indicate list/comparison queries
    LIST_KEYWORDS = ['all', 'list', 'which courses', 'what courses', 'show me', 'give me']
    COMPARISON_KEYWORDS = ['compare', 'difference', 'versus', 'vs', 'between']
    COMPULSORY_KEYWORDS = ['compulsory', 'mandatory', 'required', 'must take', 'have to']
    
    @classmethod
    def extract_course_codes(cls, query: str) -> List[str]:
        """Extract course codes from query.
        
        Args:
            query: User query string
            
        Returns:
            List of course codes found (uppercase)
        """
        matches = cls.COURSE_CODE_PATTERN.findall(query.upper())
        return list(set(matches))  # Remove duplicates
    
    @classmethod
    def analyze_query(cls, query: str) -> Dict[str, any]:
        """Analyze query to determine type and parameters.
        
        Args:
            query: User query string
            
        Returns:
            Dictionary with analysis results:
            - course_codes: List of course codes found
            - query_type: 'specific', 'list', 'comparison', 'general'
            - is_compulsory_query: Whether query is about compulsory courses
            - suggested_k: Suggested number of documents to retrieve
        """
        query_lower = query.lower()
        course_codes = cls.extract_course_codes(query)
        
        # Determine query type
        is_compulsory_query = any(keyword in query_lower for keyword in cls.COMPULSORY_KEYWORDS)
        is_list_query = any(keyword in query_lower for keyword in cls.LIST_KEYWORDS)
        is_comparison_query = any(keyword in query_lower for keyword in cls.COMPARISON_KEYWORDS)
        
        # Determine query type
        if course_codes:
            if len(course_codes) == 1:
                query_type = 'specific'
            else:
                query_type = 'comparison'
        elif is_comparison_query:
            query_type = 'comparison'
        elif is_list_query or is_compulsory_query:
            query_type = 'list'
        else:
            query_type = 'general'
        
        # Determine suggested K
        if query_type == 'specific':
            suggested_k = 1  # Only need the specific course
        elif query_type == 'comparison':
            suggested_k = min(len(course_codes) * 2, 10) if course_codes else 5
        elif query_type == 'list' or is_compulsory_query:
            suggested_k = 20  # Need many courses for lists
        else:
            suggested_k = 3  # Default
        
        return {
            'course_codes': course_codes,
            'query_type': query_type,
            'is_compulsory_query': is_compulsory_query,
            'suggested_k': suggested_k
        }


class SmartRetriever:
    """Smart retriever with dynamic K and metadata filtering."""
    
    def __init__(
        self,
        vector_store: Chroma,
        base_k: int = 3,
        max_k: int = 30,
        min_k: int = 1
    ):
        """Initialize smart retriever.
        
        Args:
            vector_store: Chroma vector store instance
            base_k: Default number of documents to retrieve
            max_k: Maximum number of documents to retrieve
            min_k: Minimum number of documents to retrieve
        """
        self.vector_store = vector_store
        self.base_k = base_k
        self.max_k = max_k
        self.min_k = min_k
        self.analyzer = QueryAnalyzer()
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter_dict: Optional[Dict] = None
    ) -> List[Document]:
        """Retrieve documents with optional filtering.
        
        Args:
            query: Query string
            k: Number of documents to retrieve (if None, uses dynamic K)
            filter_dict: Optional metadata filter dictionary
            
        Returns:
            List of retrieved documents
        """
        # Analyze query if k not provided
        if k is None:
            analysis = self.analyzer.analyze_query(query)
            k = analysis['suggested_k']
            k = max(self.min_k, min(k, self.max_k))
        
        # Use metadata filtering if course codes found
        if filter_dict is None:
            analysis = self.analyzer.analyze_query(query)
            course_codes = analysis['course_codes']
            
            # If specific course codes found, filter by them
            if course_codes and analysis['query_type'] == 'specific':
                filter_dict = {'course_code': {'$in': course_codes}}
            # For compulsory queries, we increase K but still use semantic search
            # to find courses that mention compulsory in their content
        
        # Perform retrieval
        if filter_dict:
            # Use metadata filtering - ChromaDB filter syntax: {'field': 'value'}
            # For multiple values, we'll use post-filtering
            try:
                # Try direct filter if single value
                if 'course_code' in filter_dict and isinstance(filter_dict['course_code'], dict):
                    # Handle $in operator by post-filtering
                    course_codes = filter_dict['course_code'].get('$in', [])
                    if course_codes:
                        # Get more results and filter
                        results = self.vector_store.similarity_search(query, k=k * 3)
                        # Post-filter by course_code
                        filtered_results = [
                            doc for doc in results 
                            if doc.metadata.get('course_code', '').upper() in [c.upper() for c in course_codes]
                        ]
                        results = filtered_results[:k]
                    else:
                        results = self.vector_store.similarity_search(query, k=k)
                else:
                    # Direct filter for single value
                    results = self.vector_store.similarity_search(
                        query,
                        k=k,
                        filter=filter_dict
                    )
            except Exception as e:
                # Fallback to standard search if filter fails
                print(f"[WARNING] Filter failed, using standard search: {e}")
                results = self.vector_store.similarity_search(query, k=k)
        else:
            # Standard semantic search
            results = self.vector_store.similarity_search(query, k=k)
        
        return results
    
    def retrieve_with_scores(
        self,
        query: str,
        k: Optional[int] = None,
        filter_dict: Optional[Dict] = None
    ) -> List[Tuple[Document, float]]:
        """Retrieve documents with similarity scores.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            filter_dict: Optional metadata filter dictionary
            
        Returns:
            List of tuples (Document, score)
        """
        if k is None:
            analysis = self.analyzer.analyze_query(query)
            k = analysis['suggested_k']
            k = max(self.min_k, min(k, self.max_k))
        
        if filter_dict is None:
            analysis = self.analyzer.analyze_query(query)
            course_codes = analysis['course_codes']
            if course_codes and analysis['query_type'] == 'specific':
                filter_dict = {'course_code': {'$in': course_codes}}
        
        if filter_dict:
            try:
                # Handle filter similar to retrieve method
                if 'course_code' in filter_dict and isinstance(filter_dict['course_code'], dict):
                    course_codes = filter_dict['course_code'].get('$in', [])
                    if course_codes:
                        results = self.vector_store.similarity_search_with_score(query, k=k * 3)
                        filtered_results = [
                            (doc, score) for doc, score in results 
                            if doc.metadata.get('course_code', '').upper() in [c.upper() for c in course_codes]
                        ]
                        results = filtered_results[:k]
                    else:
                        results = self.vector_store.similarity_search_with_score(query, k=k)
                else:
                    results = self.vector_store.similarity_search_with_score(
                        query,
                        k=k,
                        filter=filter_dict
                    )
            except Exception as e:
                print(f"[WARNING] Filter failed, using standard search: {e}")
                results = self.vector_store.similarity_search_with_score(query, k=k)
        else:
            results = self.vector_store.similarity_search_with_score(query, k=k)
        
        return results


class SmartRetrieverWrapper(BaseRetriever):
    """LangChain-compatible wrapper for SmartRetriever."""
    
    def __init__(self, smart_retriever: SmartRetriever):
        """Initialize wrapper.
        
        Args:
            smart_retriever: SmartRetriever instance
        """
        super().__init__()
        self.smart_retriever = smart_retriever
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents for query.
        
        Args:
            query: Query string
            
        Returns:
            List of relevant documents
        """
        return self.smart_retriever.retrieve(query)
    
    def invoke(self, query: str, *args, **kwargs) -> List[Document]:
        """Invoke retriever (LangChain interface).
        
        Args:
            query: Query string
            
        Returns:
            List of documents
        """
        return self._get_relevant_documents(query)

