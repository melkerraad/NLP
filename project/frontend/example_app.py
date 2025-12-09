"""Example Streamlit frontend application.

This is a template showing how to create a simple chat interface.
"""

import streamlit as st
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import your modules (adjust based on your actual implementation)
# from retrieval.example_retrieval import CourseRetriever
# from generation.example_generation import RAGGenerator


def main():
    """Main Streamlit app."""
    st.title("University Course Chatbot")
    st.markdown("Ask me anything about university courses!")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize components (uncomment when ready)
    # if "retriever" not in st.session_state:
    #     st.session_state.retriever = CourseRetriever()
    # if "generator" not in st.session_state:
    #     st.session_state.generator = RAGGenerator()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                st.caption(f"Sources: {', '.join(message['sources'])}")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about courses..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response (placeholder for now)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # TODO: Replace with actual RAG pipeline
                # 1. Retrieve relevant documents
                # retrieved_docs = st.session_state.retriever.retrieve(prompt)
                # 2. Generate response
                # result = st.session_state.generator.generate_with_sources(
                #     prompt, retrieved_docs
                # )
                
                # Placeholder response
                response = "This is a placeholder response. Implement the RAG pipeline to generate actual answers."
                sources = []
                
                st.markdown(response)
                if sources:
                    st.caption(f"Sources: {', '.join(sources)}")
        
        # Add assistant response to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources
        })


if __name__ == "__main__":
    main()

