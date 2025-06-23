import streamlit as st
import os
from src.document_processor import DocumentProcessor
from src.rag_pipeline import RAGPipeline
from src.utils import (
    display_chat_message, 
    create_confidence_chart, 
    format_sources, 
    validate_file_upload,
    get_system_stats
)
from dotenv import load_dotenv
load_dotenv()


# Page configuration
st.set_page_config(
    page_title="DocuMind - RAG Q&A System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stats-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        color: #333;
    }
    .source-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border-left: 3px solid #28a745;
        margin: 0.5rem 0;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline()
    if 'doc_processor' not in st.session_state:
        st.session_state.doc_processor = DocumentProcessor()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []

def main():
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 DocuMind - Intelligent Document Q&A</h1>
        <p>Upload documents and ask questions powered by RAG technology</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Document Management")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['pdf', 'txt', 'docx'],
            accept_multiple_files=True,
            help="Upload PDF, TXT, or DOCX files to create your knowledge base"
        )
        
        # Process uploaded files
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.processed_files:
                    is_valid, message = validate_file_upload(uploaded_file)
                    
                    if is_valid:
                        with st.spinner(f"Processing {uploaded_file.name}..."):
                            try:
                                # Extract text
                                text = st.session_state.doc_processor.process_document(
                                    uploaded_file.getvalue(), 
                                    uploaded_file.name
                                )
                                
                                # Chunk text
                                chunks = st.session_state.doc_processor.chunk_text(text)
                                
                                # Add to vector store
                                metadata = [{"filename": uploaded_file.name, "chunk_id": i} 
                                          for i in range(len(chunks))]
                                st.session_state.rag_pipeline.add_documents(chunks, metadata)
                                
                                st.session_state.processed_files.append(uploaded_file.name)
                                st.success(f"✅ {uploaded_file.name} processed successfully!")
                                
                            except Exception as e:
                                st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                    else:
                        st.error(message)
        
        # System statistics
        if st.session_state.processed_files:
            st.header("📊 System Stats")
            stats = get_system_stats(st.session_state.rag_pipeline.vector_store)
            
            for key, value in stats.items():
                st.markdown(f"""
                <div class="stats-card">
                    <strong>{key}:</strong> {value}
                </div>
                """, unsafe_allow_html=True)

        # # Connection test button
        # if st.button(".TestCheck API Connection"):
        #     connection_status = st.session_state.rag_pipeline.test_connection()
        #     if connection_status["status"] == "success":
        #         st.success(f"🎉 Connection successful! Response: {connection_status['response']}")
        #     else:
        #         st.error(f"❌ Connection failed: {connection_status['message']}")
        
        # Clear data button
        if st.button("🗑️ Clear All Data", type="secondary"):
            st.session_state.rag_pipeline = RAGPipeline()
            st.session_state.chat_history = []
            st.session_state.processed_files = []
            st.success("All data cleared!")
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Display chat history
        for message in st.session_state.chat_history:
            display_chat_message(message["role"], message["content"])
        
        # Query input
        query = st.text_input("Ask a question about your documents...")
        if query :
            if not st.session_state.processed_files:
                st.warning("Please upload some documents first!")
            else:
                # Add user message to chat
                st.session_state.chat_history.append({"role": "user", "content": query})
                display_chat_message("user", query)
                
                # Get RAG response
                with st.spinner("Thinking..."):
                    response = st.session_state.rag_pipeline.query(query)
                
                # Display assistant response
                assistant_message = response["answer"]
                st.session_state.chat_history.append({"role": "assistant", "content": assistant_message})
                display_chat_message("assistant", assistant_message)
                
                # Store response for sidebar display
                st.session_state.last_response = response
    
    with col2:
        st.header("📈 Response Analysis")
        
        if hasattr(st.session_state, 'last_response'):
            response = st.session_state.last_response
            
            # Confidence chart
            fig = create_confidence_chart(response["confidence"])
            st.plotly_chart(fig, use_container_width=True)
            
            # Response metrics
            st.metric("Sources Used", response["source_count"])
            st.metric("Confidence Score", f"{response['confidence']:.2f}")
            
            # Source documents
            if response["sources"]:
                st.subheader("📚 Source Documents")
                # formatted_sources = format_sources(response["sources"])
                formatted_sources = format_sources(response["sources"], max_length=None)

                
                for source in formatted_sources:
                    st.markdown(f"""
                    <div class="source-box">
                        {source}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Ask a question to see response analysis here!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Built with ❤️ using Streamlit, Hugging Face, and FAISS</p>
        <p>📧 Contact: kajaani1705@gmail.com </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
