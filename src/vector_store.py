import numpy as np
import faiss
import pickle
import os
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer

class VectorStore:
    """FAISS-based vector store for document embeddings"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.texts = []
        self.metadata = []
    
    def add_documents(self, texts: List[str], metadata: List[Dict] = None):
        """Add documents to the vector store"""
        if metadata is None:
            metadata = [{"index": i} for i in range(len(texts))]
        
        # Generate embeddings
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        
        # Normalize embeddings for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store texts and metadata
        self.texts.extend(texts)
        self.metadata.extend(metadata)
    
    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Search for similar documents"""
        if self.index.ntotal == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.model.encode([query], convert_to_tensor=False)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.texts):
                results.append((self.texts[idx], float(score), self.metadata[idx]))
        
        return results
    
    def save(self, filepath: str):
        """Save vector store to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.faiss")
        
        # Save texts and metadata
        with open(f"{filepath}.pkl", "wb") as f:
            pickle.dump({
                "texts": self.texts,
                "metadata": self.metadata,
                "dimension": self.dimension
            }, f)
    
    def load(self, filepath: str):
        """Load vector store from disk"""
        # Load FAISS index
        self.index = faiss.read_index(f"{filepath}.faiss")
        
        # Load texts and metadata
        with open(f"{filepath}.pkl", "rb") as f:
            data = pickle.load(f)
            self.texts = data["texts"]
            self.metadata = data["metadata"]
            self.dimension = data["dimension"]