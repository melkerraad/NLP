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
    
    # Section-specific keywords for query enhancement
    PREREQUISITE_KEYWORDS = ['prerequisite', 'prerequisit', 'required course', 'required courses', 'what do i need']
    LEARNING_OUTCOME_KEYWORDS = ['learning outcome', 'learning outcomes', 'what will i learn', 'what can i do']
    EXAM_KEYWORDS = ['examination', 'exam', 'exams', 'how is it examined', 'grading', 'assessment']
    CONTENT_KEYWORDS = ['content', 'what is covered', 'what topics', 'syllabus', 'curriculum']
    ORGANISATION_KEYWORDS = ['organisation', 'organization', 'how is it organized', 'structure', 'format']
    
    @classmethod
    def rewrite_query(cls, query: str) -> List[str]:
        """Generate query variations to improve retrieval recall.
        
        Creates multiple query variations based on the original query to capture
        different phrasings and improve retrieval coverage.
        
        Args:
            query: Original user query
            
        Returns:
            List of query variations (including original)
        """
        query_lower = query.lower()
        variations = [query]  # Always include original
        
        # Extract course codes if present
        course_codes = cls.extract_course_codes(query)
        course_code_str = ' '.join(course_codes) if course_codes else ''
        
        # Generate variations based on query type and keywords
        if any(keyword in query_lower for keyword in cls.PREREQUISITE_KEYWORDS):
            # Prerequisite query variations
            if course_codes:
                variations.extend([
                    f"prerequisites {course_code_str}",
                    f"required courses {course_code_str}",
                    f"what courses needed {course_code_str}",
                    f"entry requirements {course_code_str}"
                ])
            else:
                variations.extend([
                    "prerequisites",
                    "required courses",
                    "entry requirements"
                ])
        
        if any(keyword in query_lower for keyword in cls.EXAM_KEYWORDS):
            # Exam query variations
            if course_codes:
                variations.extend([
                    f"examination {course_code_str}",
                    f"how is {course_code_str} examined",
                    f"grading {course_code_str}",
                    f"assessment {course_code_str}"
                ])
            else:
                variations.extend([
                    "examination",
                    "how is course examined",
                    "grading assessment"
                ])
        
        if any(keyword in query_lower for keyword in cls.LEARNING_OUTCOME_KEYWORDS):
            # Learning outcome variations
            if course_codes:
                variations.extend([
                    f"learning outcomes {course_code_str}",
                    f"what will I learn {course_code_str}",
                    f"course objectives {course_code_str}"
                ])
            else:
                variations.extend([
                    "learning outcomes",
                    "what will I learn",
                    "course objectives"
                ])
        
        if any(keyword in query_lower for keyword in cls.CONTENT_KEYWORDS):
            # Content query variations
            if course_codes:
                variations.extend([
                    f"content {course_code_str}",
                    f"topics covered {course_code_str}",
                    f"syllabus {course_code_str}"
                ])
            else:
                variations.extend([
                    "course content",
                    "topics covered",
                    "syllabus"
                ])
        
        # General query expansion: add simpler versions
        if course_codes and len(variations) == 1:
            # If no specific variations, create general ones
            variations.append(course_code_str)
            variations.append(f"information about {course_code_str}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variations = []
        for v in variations:
            v_lower = v.lower().strip()
            if v_lower and v_lower not in seen:
                seen.add(v_lower)
                unique_variations.append(v)
        
        return unique_variations[:5]  # Limit to 5 variations
    
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
        
        # Detect section-specific queries for query enhancement
        is_prerequisite_query = any(keyword in query_lower for keyword in cls.PREREQUISITE_KEYWORDS)
        is_learning_outcome_query = any(keyword in query_lower for keyword in cls.LEARNING_OUTCOME_KEYWORDS)
        is_exam_query = any(keyword in query_lower for keyword in cls.EXAM_KEYWORDS)
        is_content_query = any(keyword in query_lower for keyword in cls.CONTENT_KEYWORDS)
        is_organisation_query = any(keyword in query_lower for keyword in cls.ORGANISATION_KEYWORDS)
        
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
        
        # Determine suggested K (increased to 5 for better coverage)
        # Increase k further for section-specific queries to ensure relevant sections are retrieved
        base_k = 5
        if query_type == 'specific':
            if is_prerequisite_query or is_learning_outcome_query or is_exam_query:
                suggested_k = 8  # More chunks for section-specific queries
            else:
                suggested_k = base_k
        elif query_type == 'comparison':
            suggested_k = base_k
        elif query_type == 'list' or is_compulsory_query:
            suggested_k = base_k
        else:
            suggested_k = base_k
        
        return {
            'course_codes': course_codes,
            'query_type': query_type,
            'is_compulsory_query': is_compulsory_query,
            'is_prerequisite_query': is_prerequisite_query,
            'is_learning_outcome_query': is_learning_outcome_query,
            'is_exam_query': is_exam_query,
            'is_content_query': is_content_query,
            'is_organisation_query': is_organisation_query,
            'suggested_k': suggested_k
        }


class SmartRetriever:
    """Smart retriever with dynamic K and metadata filtering."""
    
    def __init__(
        self,
        vector_store: Chroma,
        base_k: int = 5,
        max_k: int = 30,
        min_k: int = 1,
        max_similarity_distance: float = 0.8,
        use_query_rewriting: bool = True
    ):
        """Initialize smart retriever.
        
        Args:
            vector_store: Chroma vector store instance
            base_k: Default number of documents to retrieve
            max_k: Maximum number of documents to retrieve
            min_k: Minimum number of documents to retrieve
            max_similarity_distance: Maximum similarity distance threshold (lower = more strict)
            use_query_rewriting: Whether to use query rewriting/expansion
        """
        self.vector_store = vector_store
        self.base_k = base_k
        self.max_k = max_k
        self.min_k = min_k
        self.max_similarity_distance = max_similarity_distance
        self.use_query_rewriting = use_query_rewriting
        self.analyzer = QueryAnalyzer()
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter_dict: Optional[Dict] = None
    ) -> Tuple[List[Document], List[float], bool]:
        """Retrieve documents with optional filtering.
        
        Args:
            query: Query string
            k: Number of documents to retrieve (if None, uses dynamic K)
            filter_dict: Optional metadata filter dictionary
            
        Returns:
            Tuple of (retrieved documents, similarity scores, has_good_info)
            has_good_info: True if at least one result is below threshold
        """
        # Use query rewriting if enabled
        if self.use_query_rewriting:
            query_variations = self.analyzer.rewrite_query(query)
            # Use the first variation (original) for now, could merge results from all
            query = query_variations[0]
        
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
        
        # Analyze query for section-specific keywords
        analysis = self.analyzer.analyze_query(query)
        is_prerequisite_query = analysis.get('is_prerequisite_query', False)
        is_learning_outcome_query = analysis.get('is_learning_outcome_query', False)
        is_exam_query = analysis.get('is_exam_query', False)
        is_content_query = analysis.get('is_content_query', False)
        is_organisation_query = analysis.get('is_organisation_query', False)
        
        # Use similarity search with scores for proper re-ranking
        use_score_based_boost = (is_prerequisite_query or is_learning_outcome_query or 
                                 is_exam_query or is_content_query or is_organisation_query)
        
        # Perform retrieval with hybrid search (semantic + keyword boosting)
        if filter_dict:
            # Use metadata filtering - ChromaDB filter syntax: {'field': 'value'}
            try:
                # Handle course_code filtering (for chunked documents)
                if 'course_code' in filter_dict and isinstance(filter_dict['course_code'], dict):
                    # Handle $in operator - get chunks from specific courses
                    course_codes = filter_dict['course_code'].get('$in', [])
                    if course_codes:
                        # Use ChromaDB metadata filter to get chunks from specific courses
                        # This ensures we get chunks from the right course(s)
                        try:
                            # Retrieve more documents for hybrid search (semantic + keyword matching)
                            # Get 2x k for semantic search, then boost/reorder by keywords
                            search_k = k * 2 if use_score_based_boost else k
                            
                            if use_score_based_boost:
                            # Use similarity_search_with_score to get scores for re-ranking
                            results_with_scores = self.vector_store.similarity_search_with_score(
                                query,
                                k=search_k,
                                filter={'course_code': {'$in': course_codes}}
                            )
                            # Filter by relevance threshold
                            results_with_scores = self._filter_by_relevance(results_with_scores)
                            # Boost matching sections and re-rank
                            results, scores = self._boost_and_rerank_with_scores(
                                results_with_scores,
                                is_prerequisite_query,
                                is_learning_outcome_query,
                                is_exam_query,
                                is_content_query,
                                is_organisation_query
                            )
                            results = results[:k]
                            scores = scores[:k]
                            has_good_info = any(score <= self.max_similarity_distance for score in scores) if scores else False
                            return results, scores, has_good_info
                        else:
                            results_with_scores = self.vector_store.similarity_search_with_score(
                                query,
                                k=search_k,
                                filter={'course_code': {'$in': course_codes}}
                            )
                            results_with_scores = self._filter_by_relevance(results_with_scores)
                            results = [doc for doc, _ in results_with_scores[:k]]
                            scores = [score for _, score in results_with_scores[:k]]
                            has_good_info = any(score <= self.max_similarity_distance for score in scores) if scores else False
                            return results, scores, has_good_info
                        except Exception as filter_error:
                            # Fallback: post-filter after getting more results
                            print(f"[INFO] Using post-filtering fallback: {filter_error}")
                            search_k = k * 2 if use_score_based_boost else k
                            if use_score_based_boost:
                                results_with_scores = self.vector_store.similarity_search_with_score(query, k=search_k * 2)
                                # Post-filter by course_code
                                filtered_results_with_scores = [
                                    (doc, score) for doc, score in results_with_scores
                                    if doc.metadata.get('course_code', '').upper() in [c.upper() for c in course_codes]
                                ]
                                # Filter by relevance threshold
                                filtered_results_with_scores = self._filter_by_relevance(filtered_results_with_scores)
                                # Boost and re-rank
                                results, scores = self._boost_and_rerank_with_scores(
                                    filtered_results_with_scores,
                                    is_prerequisite_query,
                                    is_learning_outcome_query,
                                    is_exam_query,
                                    is_content_query,
                                    is_organisation_query
                                )
                                results = results[:k]
                                scores = scores[:k]
                                has_good_info = any(score <= self.max_similarity_distance for score in scores) if scores else False
                                return results, scores, has_good_info
                            else:
                                results_with_scores = self.vector_store.similarity_search_with_score(query, k=search_k * 2)
                                # Post-filter by course_code
                                filtered_results_with_scores = [
                                    (doc, score) for doc, score in results_with_scores
                                    if doc.metadata.get('course_code', '').upper() in [c.upper() for c in course_codes]
                                ]
                                filtered_results_with_scores = self._filter_by_relevance(filtered_results_with_scores)
                                results = [doc for doc, _ in filtered_results_with_scores[:k]]
                                scores = [score for _, score in filtered_results_with_scores[:k]]
                                has_good_info = any(score <= self.max_similarity_distance for score in scores) if scores else False
                                return results, scores, has_good_info
                    else:
                        results_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
                        results_with_scores = self._filter_by_relevance(results_with_scores)
                        results = [doc for doc, _ in results_with_scores[:k]]
                        scores = [score for _, score in results_with_scores[:k]]
                        has_good_info = any(score <= self.max_similarity_distance for score in scores) if scores else False
                        return results, scores, has_good_info
                else:
                    # Direct filter for single value
                    results_with_scores = self.vector_store.similarity_search_with_score(
                        query,
                        k=k,
                        filter=filter_dict
                    )
                    results_with_scores = self._filter_by_relevance(results_with_scores)
                    results = [doc for doc, _ in results_with_scores[:k]]
                    scores = [score for _, score in results_with_scores[:k]]
                    has_good_info = any(score <= self.max_similarity_distance for score in scores) if scores else False
                    return results, scores, has_good_info
            except Exception as e:
                # Fallback to standard search if filter fails
                print(f"[WARNING] Filter failed, using standard search: {e}")
                results_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
                results_with_scores = self._filter_by_relevance(results_with_scores)
                results = [doc for doc, _ in results_with_scores[:k]]
                scores = [score for _, score in results_with_scores[:k]]
                has_good_info = any(score <= self.max_similarity_distance for score in scores) if scores else False
                return results, scores, has_good_info
        else:
            # Standard semantic search with keyword boosting
            if use_score_based_boost:
                search_k = k * 2
                results_with_scores = self.vector_store.similarity_search_with_score(query, k=search_k)
                # Filter by relevance threshold
                results_with_scores = self._filter_by_relevance(results_with_scores)
                # Boost and re-rank based on scores
                results, scores = self._boost_and_rerank_with_scores(
                    results_with_scores,
                    is_prerequisite_query,
                    is_learning_outcome_query,
                    is_exam_query,
                    is_content_query,
                    is_organisation_query
                )
                results = results[:k]
                scores = scores[:k]
            else:
                # For non-boosted queries, still filter by relevance if we have scores
                results_with_scores = self.vector_store.similarity_search_with_score(query, k=k * 2)
                results_with_scores = self._filter_by_relevance(results_with_scores)
                results = [doc for doc, _ in results_with_scores[:k]]
                scores = [score for _, score in results_with_scores[:k]]
            
            # Check if we have at least one good result (score below threshold)
            has_good_info = any(score <= self.max_similarity_distance for score in scores) if scores else False
            return results, scores, has_good_info
    
    def _boost_relevant_sections(
        self,
        results: List[Document],
        is_prerequisite_query: bool,
        is_learning_outcome_query: bool,
        is_exam_query: bool,
        is_content_query: bool,
        is_organisation_query: bool
    ) -> List[Document]:
        """Boost documents with matching section names based on query keywords.
        
        This implements hybrid search by combining semantic similarity with keyword matching.
        
        Args:
            results: List of retrieved documents
            is_prerequisite_query: Whether query mentions prerequisites
            is_learning_outcome_query: Whether query mentions learning outcomes
            is_exam_query: Whether query mentions exams
            is_content_query: Whether query mentions content
            is_organisation_query: Whether query mentions organisation
            
        Returns:
            Reordered list with matching sections boosted to the top
        """
        if not (is_prerequisite_query or is_learning_outcome_query or 
                is_exam_query or is_content_query or is_organisation_query):
            return results
        
        # Separate documents into matching and non-matching
        matching_docs = []
        non_matching_docs = []
        
        for doc in results:
            section_name = doc.metadata.get('section_name', '').lower()
            section_name_original = doc.metadata.get('section_name_original', '').lower()
            chunk_type = doc.metadata.get('chunk_type', '')
            
            is_match = False
            
            # Check if section matches query keywords (can match multiple keywords)
            if is_prerequisite_query:
                if 'prerequisite' in section_name or 'prerequisite' in section_name_original:
                    is_match = True
            if is_learning_outcome_query:
                if 'learning outcome' in section_name or 'learning outcome' in section_name_original:
                    is_match = True
            if is_exam_query:
                if 'examination' in section_name or 'exam' in section_name or 'examination' in section_name_original:
                    is_match = True
            if is_content_query:
                if 'content' in section_name or 'content' in section_name_original:
                    is_match = True
            if is_organisation_query:
                if 'organisation' in section_name or 'organization' in section_name or \
                   'organisation' in section_name_original or 'organization' in section_name_original:
                    is_match = True
            
            if is_match:
                matching_docs.append(doc)
            else:
                non_matching_docs.append(doc)
        
        # Return matching docs first, then non-matching (preserving semantic order within each group)
        return matching_docs + non_matching_docs
    
    def _filter_by_relevance(
        self,
        results_with_scores: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        """Filter results by similarity threshold.
        
        Removes documents with similarity scores above the threshold,
        indicating they are not relevant enough.
        
        Args:
            results_with_scores: List of (Document, score) tuples
            
        Returns:
            Filtered list of (Document, score) tuples
        """
        filtered = [
            (doc, score) for doc, score in results_with_scores
            if score <= self.max_similarity_distance
        ]
        return filtered
    
    def _boost_and_rerank_with_scores(
        self,
        results_with_scores: List[Tuple[Document, float]],
        is_prerequisite_query: bool,
        is_learning_outcome_query: bool,
        is_exam_query: bool,
        is_content_query: bool,
        is_organisation_query: bool
    ) -> Tuple[List[Document], List[float]]:
        """Boost and re-rank documents using similarity scores + keyword matching.
        
        This provides stronger boosting than simple reordering by actually adjusting scores.
        
        Args:
            results_with_scores: List of (Document, score) tuples
            is_prerequisite_query: Whether query mentions prerequisites
            is_learning_outcome_query: Whether query mentions learning outcomes
            is_exam_query: Whether query mentions exams
            is_content_query: Whether query mentions content
            is_organisation_query: Whether query mentions organisation
            
        Returns:
            Tuple of (re-ranked documents, scores)
        """
        if not (is_prerequisite_query or is_learning_outcome_query or 
                is_exam_query or is_content_query or is_organisation_query):
            docs = [doc for doc, _ in results_with_scores]
            scores = [score for _, score in results_with_scores]
            return docs, scores
        
        # Calculate boosted scores
        boosted_results = []
        for doc, score in results_with_scores:
            section_name = doc.metadata.get('section_name', '').lower()
            section_name_original = doc.metadata.get('section_name_original', '').lower()
            
            # Check if section matches query keywords
            is_match = False
            if is_prerequisite_query:
                if 'prerequisite' in section_name or 'prerequisite' in section_name_original:
                    is_match = True
            if is_learning_outcome_query:
                if 'learning outcome' in section_name or 'learning outcome' in section_name_original:
                    is_match = True
            if is_exam_query:
                if 'examination' in section_name or 'exam' in section_name or 'examination' in section_name_original:
                    is_match = True
            if is_content_query:
                if 'content' in section_name or 'content' in section_name_original:
                    is_match = True
            if is_organisation_query:
                if 'organisation' in section_name or 'organization' in section_name or \
                   'organisation' in section_name_original or 'organization' in section_name_original:
                    is_match = True
            
            # Boost matching sections: add a significant boost to their scores
            # Lower scores = more similar, so we subtract to boost (make score lower = better rank)
            if is_match:
                # Strong boost: reduce score by 0.3 (makes it rank much higher)
                boosted_score = score - 0.3
            else:
                boosted_score = score
            
            boosted_results.append((doc, boosted_score))
        
        # Sort by boosted score (lower = better)
        boosted_results.sort(key=lambda x: x[1])
        
        # Return documents and scores separately
        docs = [doc for doc, _ in boosted_results]
        scores = [score for _, score in boosted_results]
        return docs, scores
    
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
                        # Try using ChromaDB metadata filter
                        try:
                            results = self.vector_store.similarity_search_with_score(
                                query,
                                k=k,
                                filter={'course_code': {'$in': course_codes}}
                            )
                        except Exception as filter_error:
                            # Fallback: post-filter after getting more results
                            print(f"[INFO] Using post-filtering fallback: {filter_error}")
                            results = self.vector_store.similarity_search_with_score(query, k=k * 2)
                            filtered_results = [
                                (doc, score) for doc, score in results 
                                if doc.metadata.get('course_code', '').upper() in [c.upper() for c in course_codes]
                            ]
                            results = filtered_results[:k]
                        
                        # Prioritize metadata chunks
                        metadata_chunks = [(doc, score) for doc, score in results if doc.metadata.get('chunk_type') == 'metadata']
                        description_chunks = [(doc, score) for doc, score in results if doc.metadata.get('chunk_type') == 'description']
                        
                        if metadata_chunks:
                            results = metadata_chunks + description_chunks[:k - len(metadata_chunks)]
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
    
    smart_retriever: SmartRetriever  # Declare as a Pydantic field
    
    def __init__(self, smart_retriever: SmartRetriever):
        """Initialize wrapper.
        
        Args:
            smart_retriever: SmartRetriever instance
        """
        super().__init__(smart_retriever=smart_retriever)
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents for query.
        
        Args:
            query: Query string
            
        Returns:
            List of relevant documents
        """
        docs, _, _ = self.smart_retriever.retrieve(query)
        return docs
    
    def retrieve_with_metadata(self, query: str) -> Tuple[List[Document], List[float], bool]:
        """Retrieve documents with scores and information quality flag.
        
        Args:
            query: Query string
            
        Returns:
            Tuple of (documents, scores, has_good_info)
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
    
    # Store metadata for access by RAG chain
    _last_scores = []
    _last_has_good_info = False

