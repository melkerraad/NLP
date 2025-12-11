"""Gradio UI for RAG chatbot."""

import sys
import gradio as gr
import torch
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import load_config
from src.retrieval.vector_store import VectorStoreFactory
from src.generation.llm_factory import create_llm_from_config
from src.generation.rag_chain import create_rag_chain


class RAGChatbotUI:
    """Gradio UI wrapper for RAG chatbot."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize UI with configuration.
        
        Args:
            config_path: Optional path to config file
        """
        print("Loading configuration...")
        self.config = load_config(config_path)
        
        print("Initializing RAG components...")
        self._initialize_components()
        
        print("[OK] UI components initialized")
        self.tracker = TimingTracker()
    
    def _initialize_components(self):
        """Initialize RAG components (vector store, LLM, chain)."""
        # Load vector store
        vector_config = self.config.get_vector_store_config()
        retrieval_config = self.config.get_retrieval_config()
        project_root = Path(__file__).parent.parent.parent
        
        persist_dir = str(project_root / vector_config.get("persist_directory", "data/chroma_db"))
        
        # Auto-detect device for embeddings
        embedding_device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("Loading vector store...")
        if torch.cuda.is_available():
            print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("[INFO] Using CPU for embeddings")
        
        self.vector_store = VectorStoreFactory.load_existing(
            collection_name=vector_config.get("collection_name", "chalmers_courses"),
            persist_directory=persist_dir,
            embedding_model=retrieval_config.get("embedding_model", "all-MiniLM-L6-v2"),
            device=embedding_device
        )
        
        # Create retriever
        top_k = retrieval_config.get("top_k", 3)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})
        
        # Create LLM
        print("Loading LLM...")
        self.llm = create_llm_from_config(self.config.config)
        
        # Create RAG chain
        print("Building RAG chain...")
        self.rag_chain = create_rag_chain(
            retriever=self.retriever,
            llm=self.llm,
            template_name="rag"
        )
        
        print("[OK] RAG chain ready")
    
    def _query_rag(self, query: str) -> tuple:
        """Process query through RAG chain.
        
        Args:
            query: User query
            
        Returns:
            Tuple of (answer, sources_string)
        """
        if not query or not query.strip():
            return "Please enter a question.", ""
        
        try:
            # Time retrieval
            with self.tracker.time_operation("retrieval"):
                retrieved_docs = self.retriever.invoke(query)
            
            # Time generation
            with self.tracker.time_operation("generation"):
                result = self.rag_chain.invoke_with_sources(query)
            
            answer = result.get('answer', 'No answer generated.')
            sources = result.get('sources', [])
            
            # Get timing information
            retrieval_time = self.tracker.get_timing("retrieval")
            generation_time = self.tracker.get_timing("generation")
            
            # Format sources with timing info
            timing_info = []
            if retrieval_time:
                timing_info.append(f"Retrieval: {retrieval_time*1000:.0f}ms")
            if generation_time:
                timing_info.append(f"Generation: {generation_time*1000:.0f}ms")
            
            if sources:
                sources_str = f"Sources: {', '.join(sources)}"
                if timing_info:
                    sources_str += f" | {' | '.join(timing_info)}"
            else:
                sources_str = "No sources found."
                if timing_info:
                    sources_str += f" | {' | '.join(timing_info)}"
            
            return answer, sources_str
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg, ""
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface.
        
        Returns:
            Gradio Blocks interface
        """
        ui_config = self.config.get_ui_config()
        
        with gr.Blocks(title=ui_config.get("title", "RAG Chatbot")) as interface:
            gr.Markdown(f"# {ui_config.get('title', 'RAG Chatbot')}")
            gr.Markdown(ui_config.get("description", "Ask questions using RAG"))
            
            with gr.Row():
                with gr.Column(scale=2):
                    query_input = gr.Textbox(
                        label="Question",
                        placeholder="What courses teach deep learning?",
                        lines=3
                    )
                    submit_btn = gr.Button("Submit", variant="primary")
                
                with gr.Column(scale=1):
                    gr.Markdown("### Instructions")
                    gr.Markdown("""
                    - Ask questions about Chalmers courses
                    - The system will retrieve relevant courses and generate an answer
                    - Sources will be shown below the answer
                    """)
            
            with gr.Row():
                answer_output = gr.Textbox(
                    label="Answer",
                    lines=10,
                    interactive=False
                )
            
            with gr.Row():
                sources_output = gr.Textbox(
                    label="Sources",
                    lines=2,
                    interactive=False
                )
            
            # Connect components
            submit_btn.click(
                fn=self._query_rag,
                inputs=query_input,
                outputs=[answer_output, sources_output]
            )
            
            # Allow Enter key to submit
            query_input.submit(
                fn=self._query_rag,
                inputs=query_input,
                outputs=[answer_output, sources_output]
            )
        
        return interface
    
    def launch(self, **kwargs):
        """Launch Gradio interface.
        
        Args:
            **kwargs: Additional arguments to pass to Gradio launch
        """
        ui_config = self.config.get_ui_config()
        
        # Default launch arguments
        launch_kwargs = {
            "server_name": "0.0.0.0",  # Allow external connections
            "server_port": ui_config.get("port", 7860),
            "share": ui_config.get("share", False),
        }
        
        # Override with provided kwargs
        launch_kwargs.update(kwargs)
        
        interface = self.create_interface()
        interface.launch(**launch_kwargs)


def main():
    """Main function to launch UI."""
    print("=" * 70)
    print("Starting RAG Chatbot UI")
    print("=" * 70)
    
    ui = RAGChatbotUI()
    ui.launch()


if __name__ == "__main__":
    main()

