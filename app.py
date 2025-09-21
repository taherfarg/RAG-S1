#!/usr/bin/env python3
"""
RAG System - Vector Database Setup & Configuration
Main application for Retrieval-Augmented Generation system.
"""

import os
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

# Import required libraries for vector operations
import faiss
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch

# Import error handling for missing packages
try:
    import faiss
except ImportError:
    raise ImportError("FAISS is required for vector operations. Install with: pip install faiss-cpu")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("sentence-transformers is required for embeddings. Install with: pip install sentence-transformers")


@dataclass
class VectorDBConfig:
    """Configuration for vector database."""

    # Embedding model settings
    embedding_model_name: str = "all-MiniLM-L6-v2"  # Lightweight, fast model
    embedding_dimension: int = 384  # Dimension for the chosen model

    # Vector database settings
    index_type: str = "IndexFlatIP"  # Inner product for cosine similarity
    index_file: str = "vector_index.faiss"
    metadata_file: str = "chunks_metadata.pkl"

    # Performance settings
    batch_size: int = 32
    max_text_length: int = 512  # Max tokens for embedding

    # Search settings
    top_k: int = 5  # Number of chunks to retrieve
    similarity_threshold: float = 0.0  # Minimum similarity score

    # Query processing settings
    enable_query_expansion: bool = True
    query_expansion_factor: int = 2

    # Context management settings
    max_context_length: int = 4000  # Maximum tokens for context window
    context_overlap: int = 200  # Overlap between context chunks

    # Data paths
    chunks_file: str = "chunks/all_chunks.json"
    vector_db_dir: str = "vector_db"


@dataclass
class RAGConfig:
    """Main RAG system configuration."""

    # Vector database config
    vector_db: VectorDBConfig = None

    # Model settings
    device: str = "auto"  # Auto-detect CPU/CUDA

    # System settings
    log_level: str = "INFO"
    save_intermediate: bool = True

    def __post_init__(self):
        if self.vector_db is None:
            self.vector_db = VectorDBConfig()


class VectorDatabase:
    """Vector database for storing and retrieving text embeddings."""

    def __init__(self, config: VectorDBConfig, log_level: str = "INFO", device: str = "auto"):
        self.config = config
        self.log_level = log_level
        self.device = device
        self.embedding_model = None
        self.index = None
        self.chunks_metadata = []
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('rag_system.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )

    def initialize_embedding_model(self) -> SentenceTransformer:
        """Initialize the sentence transformer model."""
        try:
            logging.info(f"Loading embedding model: {self.config.embedding_model_name}")

            # Detect device
            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device

            model = SentenceTransformer(
                self.config.embedding_model_name,
                device=device
            )

            # Verify model dimension
            test_embedding = model.encode("test")
            actual_dim = test_embedding.shape[0]

            if actual_dim != self.config.embedding_dimension:
                logging.warning(f"Model dimension mismatch: expected {self.config.embedding_dimension}, got {actual_dim}")
                self.config.embedding_dimension = actual_dim

            logging.info(f"Model loaded successfully on {device} with dimension {actual_dim}")
            return model

        except Exception as e:
            logging.error(f"Failed to load embedding model: {str(e)}")
            raise

    def create_index(self) -> faiss.Index:
        """Create FAISS index for vector storage."""
        try:
            if self.config.index_type == "IndexFlatIP":
                index = faiss.IndexFlatIP(self.config.embedding_dimension)
            elif self.config.index_type == "IndexFlatL2":
                index = faiss.IndexFlatL2(self.config.embedding_dimension)
            else:
                raise ValueError(f"Unsupported index type: {self.config.index_type}")

            logging.info(f"Created FAISS index: {self.config.index_type}")
            return index

        except Exception as e:
            logging.error(f"Failed to create FAISS index: {str(e)}")
            raise

    def load_chunks_data(self) -> Dict[str, List[Dict]]:
        """Load chunks data from JSON file."""
        try:
            if not Path(self.config.chunks_file).exists():
                raise FileNotFoundError(f"Chunks file not found: {self.config.chunks_file}")

            with open(self.config.chunks_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            logging.info(f"Loaded chunks data with {len(data)} books")
            return data

        except Exception as e:
            logging.error(f"Failed to load chunks data: {str(e)}")
            raise

    def prepare_chunks_for_indexing(self, chunks_data: Dict[str, List[Dict]]) -> Tuple[List[str], List[Dict]]:
        """Prepare chunks for vector indexing."""
        texts = []
        metadata = []

        for book_name, chunks in chunks_data.items():
            for chunk in chunks:
                # Clean and prepare text
                text = chunk.get('content', '')
                if isinstance(text, str):
                    text = text.strip()
                else:
                    text = str(text).strip()

                if text:
                    texts.append(text)
                    # Add book information to metadata
                    chunk_metadata = chunk.get('metadata', {})
                    chunk_metadata['book_name'] = book_name
                    metadata.append(chunk_metadata)

        logging.info(f"Prepared {len(texts)} chunks for indexing")
        return texts, metadata

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for the given texts."""
        try:
            # Filter out empty or invalid texts
            valid_texts = []
            for text in texts:
                if text and isinstance(text, str) and text.strip():
                    valid_texts.append(text.strip())

            if not valid_texts:
                raise ValueError("No valid texts found for embedding generation")

            embeddings = []

            for i in range(0, len(valid_texts), self.config.batch_size):
                batch_texts = valid_texts[i:i + self.config.batch_size]

                # Truncate texts if too long and ensure they're strings
                batch_texts = [str(text)[:self.config.max_text_length] for text in batch_texts]

                # Final validation - ensure no empty strings
                batch_texts = [text for text in batch_texts if text.strip()]

                if not batch_texts:
                    continue

                try:
                    batch_embeddings = self.embedding_model.encode(
                        batch_texts,
                        show_progress_bar=True,
                        batch_size=self.config.batch_size
                    )
                    embeddings.extend(batch_embeddings)
                except Exception as batch_error:
                    logging.warning(f"Failed to encode batch {i//self.config.batch_size + 1}: {str(batch_error)}")
                    # Try encoding one by one for this batch
                    for text in batch_texts:
                        try:
                            text_embedding = self.embedding_model.encode([text])
                            embeddings.extend(text_embedding)
                        except Exception as text_error:
                            logging.warning(f"Failed to encode text: {str(text_error)}")
                            continue

                logging.info(f"Processed batch {i//self.config.batch_size + 1}/{(len(valid_texts)-1)//self.config.batch_size + 1}")

            if not embeddings:
                raise ValueError("No embeddings were generated")

            embeddings_array = np.array(embeddings, dtype=np.float32)
            logging.info(f"Generated embeddings with shape: {embeddings_array.shape}")
            return embeddings_array

        except Exception as e:
            logging.error(f"Failed to generate embeddings: {str(e)}")
            raise

    def build_index(self, chunks_data: Dict[str, List[Dict]]) -> None:
        """Build the vector index from chunks data."""
        try:
            # Prepare chunks
            texts, metadata = self.prepare_chunks_for_indexing(chunks_data)

            if not texts:
                raise ValueError("No valid texts found for indexing")

            # Generate embeddings
            embeddings = self.generate_embeddings(texts)

            # Create and populate index
            self.index = self.create_index()
            self.index.add(embeddings)

            # Store metadata
            self.chunks_metadata = metadata

            logging.info(f"Built index with {self.index.ntotal} vectors")

        except Exception as e:
            logging.error(f"Failed to build vector index: {str(e)}")
            raise

    def save_index(self) -> None:
        """Save the vector index and metadata to disk."""
        try:
            # Create vector DB directory
            vector_db_path = Path(self.config.vector_db_dir)
            vector_db_path.mkdir(exist_ok=True)

            # Save FAISS index
            index_path = vector_db_path / self.config.index_file
            faiss.write_index(self.index, str(index_path))

            # Save metadata
            metadata_path = vector_db_path / self.config.metadata_file
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.chunks_metadata, f)

            logging.info(f"Saved vector index to {index_path}")
            logging.info(f"Saved metadata to {metadata_path}")

        except Exception as e:
            logging.error(f"Failed to save vector index: {str(e)}")
            raise

    def load_index(self) -> None:
        """Load the vector index and metadata from disk."""
        try:
            vector_db_path = Path(self.config.vector_db_dir)

            # Load FAISS index
            index_path = vector_db_path / self.config.index_file
            if not index_path.exists():
                raise FileNotFoundError(f"Index file not found: {index_path}")

            self.index = faiss.read_index(str(index_path))

            # Load metadata
            metadata_path = vector_db_path / self.config.metadata_file
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

            with open(metadata_path, 'rb') as f:
                self.chunks_metadata = pickle.load(f)

            logging.info(f"Loaded vector index with {self.index.ntotal} vectors")

        except Exception as e:
            logging.error(f"Failed to load vector index: {str(e)}")
            raise

    def preprocess_query(self, query: str) -> str:
        """Preprocess and clean the query text."""
        if not query:
            return query

        # Basic text cleaning
        import re
        # Remove extra whitespace and normalize
        query = re.sub(r'\s+', ' ', query.strip())

        # Remove special characters that might cause issues
        query = re.sub(r'[^\w\s\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', ' ', query)

        return query.strip()

    def expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms or related terms."""
        # Simple query expansion - can be enhanced with thesaurus or LLM
        expanded_queries = [query]

        # Add some common variations
        if len(query.split()) > 1:
            # Add query without stop words (basic implementation)
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            meaningful_terms = [word for word in query.lower().split() if word not in stop_words]
            if meaningful_terms:
                expanded_queries.append(' '.join(meaningful_terms))

        return expanded_queries

    def search(self, query: str, top_k: Optional[int] = None,
               similarity_threshold: Optional[float] = None,
               use_expansion: bool = True) -> List[Tuple[Dict, float]]:
        """Enhanced search with query processing and multiple strategies."""
        try:
            if self.index is None or self.index.ntotal == 0:
                raise ValueError("Vector index not initialized or empty")

            k = top_k or self.config.top_k
            threshold = similarity_threshold or self.config.similarity_threshold

            # Preprocess query
            processed_query = self.preprocess_query(query)
            if not processed_query:
                logging.warning("Query became empty after preprocessing")
                return []

            logging.info(f"Processed query: '{processed_query}'")

            # Query expansion
            search_queries = [processed_query]
            if use_expansion and self.config.enable_query_expansion:
                search_queries = self.expand_query(processed_query)
                logging.info(f"Expanded to {len(search_queries)} queries")

            # Generate embeddings for all query variations
            all_results = []
            for query_text in search_queries:
                if not query_text.strip():
                    continue

                try:
                    # Generate query embedding
                    query_embedding = self.embedding_model.encode([query_text])
                    query_embedding = np.array(query_embedding, dtype=np.float32)

                    # Search
                    scores, indices = self.index.search(query_embedding, k * 2)  # Get more for reranking

                    # Format results with similarity scores
                    for score, idx in zip(scores[0], indices[0]):
                        if 0 <= idx < len(self.chunks_metadata):  # Valid index
                            metadata = self.chunks_metadata[idx].copy()
                            similarity_score = float(score)

                            # Apply similarity threshold
                            if similarity_score < threshold:
                                continue

                            # Add search metadata
                            metadata['_search'] = {
                                'similarity_score': similarity_score,
                                'query_text': query_text,
                                'original_query': query
                            }

                            all_results.append((metadata, similarity_score))

                except Exception as query_error:
                    logging.warning(f"Failed to process query variation '{query_text}': {str(query_error)}")
                    continue

            # Remove duplicates and sort by similarity score
            seen_chunks = set()
            unique_results = []

            for metadata, score in sorted(all_results, key=lambda x: x[1], reverse=True):
                # Create a unique identifier for the chunk
                chunk_id = metadata.get('chunk_id', str(metadata.get('page', 0)))
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    unique_results.append((metadata, score))

            # Limit to top_k results
            final_results = unique_results[:k]

            logging.info(f"Enhanced search returned {len(final_results)} unique results (threshold: {threshold})")
            return final_results

        except Exception as e:
            logging.error(f"Failed to perform enhanced search: {str(e)}")
            return []

    def create_context(self, search_results: List[Tuple[Dict, float]],
                      max_length: Optional[int] = None) -> str:
        """Create a coherent context from search results."""
        try:
            max_len = max_length or self.config.max_context_length

            # Sort by relevance and group by book for better context
            book_contexts = {}
            for metadata, score in search_results:
                book_name = metadata.get('book_name', 'Unknown')
                content = metadata.get('content', '')

                if book_name not in book_contexts:
                    book_contexts[book_name] = []

                book_contexts[book_name].append({
                    'content': content,
                    'score': score,
                    'page': metadata.get('page', 0),
                    'chunk_id': metadata.get('chunk_id', '')
                })

            # Build context for each book
            context_parts = []
            current_length = 0

            for book_name in sorted(book_contexts.keys()):
                chunks = book_contexts[book_name]

                # Sort chunks by page number and score
                chunks.sort(key=lambda x: (x['page'], -x['score']))

                book_context = f"\n--- From {book_name} ---\n"

                for chunk in chunks:
                    chunk_text = chunk['content'].strip()
                    if chunk_text:
                        # Add overlap handling for better context continuity
                        if context_parts and self.config.context_overlap > 0:
                            # Find overlapping content from previous chunks
                            last_chunk = context_parts[-1]
                            if len(last_chunk) > self.config.context_overlap:
                                overlap_text = last_chunk[-self.config.context_overlap:]
                                if overlap_text in chunk_text[:len(overlap_text)*2]:
                                    chunk_text = chunk_text[len(overlap_text):]

                        # Add chunk with metadata
                        chunk_with_meta = f"[Page {chunk['page']}] {chunk_text}"
                        book_context += chunk_with_meta + "\n"

                # Add book context if it fits
                if current_length + len(book_context) <= max_len:
                    context_parts.append(book_context)
                    current_length += len(book_context)
                else:
                    # Try to add partial content
                    available_space = max_len - current_length
                    if available_space > 100:  # Minimum meaningful length
                        partial_context = book_context[:available_space]
                        context_parts.append(partial_context)
                    break

            final_context = "\n".join(context_parts)

            logging.info(f"Created context with {len(final_context)} characters from {len(search_results)} results")
            return final_context

        except Exception as e:
            logging.error(f"Failed to create context: {str(e)}")
            return ""

    def conversational_search(self, query: str, conversation_history: Optional[List[Dict]] = None,
                            top_k: Optional[int] = None) -> Dict[str, Any]:
        """Perform conversational search with context awareness."""
        try:
            # Perform search
            search_results = self.search(query, top_k=top_k)

            if not search_results:
                return {
                    'query': query,
                    'context': 'No relevant information found.',
                    'results': [],
                    'total_results': 0
                }

            # Create context
            context = self.create_context(search_results)

            # Add conversation context if provided
            if conversation_history:
                # Simple conversation context - can be enhanced with more sophisticated methods
                recent_context = "\n".join([
                    f"Previous Q: {turn.get('query', '')}\nPrevious A: {turn.get('response', '')}"
                    for turn in conversation_history[-2:]  # Last 2 turns
                ])
                context = f"Conversation context:\n{recent_context}\n\nCurrent context:\n{context}"

            return {
                'query': query,
                'context': context,
                'results': search_results,
                'total_results': len(search_results)
            }

        except Exception as e:
            logging.error(f"Failed to perform conversational search: {str(e)}")
            return {
                'query': query,
                'context': f'Error: {str(e)}',
                'results': [],
                'total_results': 0
            }


def setup_rag_system(config: RAGConfig) -> VectorDatabase:
    """Setup and initialize the RAG system."""
    try:
        logging.info("Initializing RAG system...")

        # Create vector database instance
        vector_db = VectorDatabase(config.vector_db, config.log_level, config.device)

        # Initialize embedding model
        vector_db.embedding_model = vector_db.initialize_embedding_model()

        # Check if index already exists
        vector_db_path = Path(config.vector_db.vector_db_dir)
        index_path = vector_db_path / config.vector_db.index_file

        if index_path.exists():
            logging.info("Loading existing vector index...")
            vector_db.load_index()
        else:
            logging.info("Building new vector index...")
            # Load chunks data
            chunks_data = vector_db.load_chunks_data()

            # Build index
            vector_db.build_index(chunks_data)

            # Save index
            if config.save_intermediate:
                vector_db.save_index()

        logging.info("RAG system initialized successfully!")
        return vector_db

    except Exception as e:
        logging.error(f"Failed to setup RAG system: {str(e)}")
        raise


def demonstrate_search(rag_system: VectorDatabase):
    """Demonstrate the enhanced search capabilities."""
    print("\n" + "="*60)
    print("🔍 RAG System - Enhanced Query Processing Demo")
    print("="*60)

    # Test queries
    test_queries = [
        "What is artificial intelligence?",
        "Explain machine learning",
        "Tell me about deep learning",
        "How does neural networks work?",
        "What are the different types of AI?"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Query {i}: {query}")
        print("-" * 40)

        # Perform enhanced search
        results = rag_system.search(
            query,
            top_k=3,
            similarity_threshold=0.1,
            use_expansion=True
        )

        if results:
            # Create context
            context = rag_system.create_context(results, max_length=1000)

            print(f"✅ Found {len(results)} relevant chunks")
            print("📄 Context Preview:")
            print(context[:500] + "..." if len(context) > 500 else context)

            # Show top result details
            if results:
                top_result, top_score = results[0]
                print("\n🏆 Top Result:")
                print(f"   Book: {top_result.get('book_name', 'Unknown')}")
                print(f"   Page: {top_result.get('page', 'N/A')}")
                print(f"   Similarity: {top_score:.3f}")
        else:
            print("❌ No relevant results found")

        print("\n" + "-" * 60)

def interactive_mode(rag_system: VectorDatabase):
    """Interactive mode for testing queries."""
    print("\n" + "="*60)
    print("💬 RAG System - Interactive Mode")
    print("Type 'quit' or 'exit' to end the session")
    print("Type 'demo' to run the demonstration")
    print("="*60)

    conversation_history = []

    while True:
        try:
            query = input("\n❓ Enter your query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif query.lower() == 'demo':
                demonstrate_search(rag_system)
                continue
            elif not query:
                continue

            # Perform conversational search
            result = rag_system.conversational_search(
                query,
                conversation_history=conversation_history,
                top_k=5
            )

            # Display results
            print(f"\n📊 Results for: {result['query']}")
            print(f"📈 Total results: {result['total_results']}")

            if result['context']:
                print("\n📄 Context:")
                print(result['context'][:1000] + "..." if len(result['context']) > 1000 else result['context'])

            # Add to conversation history
            conversation_history.append({
                'query': query,
                'response': result['context'][:200] + "..." if len(result['context']) > 200 else result['context']
            })

            # Keep only last 5 conversations
            if len(conversation_history) > 5:
                conversation_history = conversation_history[-5:]

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def main():
    """Main execution function."""
    try:
        # Create configuration
        config = RAGConfig()

        # Setup RAG system
        rag_system = setup_rag_system(config)

        logging.info("RAG system setup completed successfully!")

        # Ask user what they want to do
        print("\n" + "="*60)
        print("🚀 RAG System Ready!")
        print("="*60)
        print("Choose an option:")
        print("1. Run demonstration (demo)")
        print("2. Interactive mode (interactive)")
        print("3. Exit")

        choice = input("\nEnter your choice (1-3): ").strip()

        if choice == '1' or choice.lower() == 'demo':
            demonstrate_search(rag_system)
        elif choice == '2' or choice.lower() == 'interactive':
            interactive_mode(rag_system)
        else:
            print("👋 Goodbye!")

    except Exception as e:
        logging.error(f"Failed to run RAG system: {str(e)}")
        raise

if __name__ == "__main__":
    main()