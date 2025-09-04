from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import os
import logging

logger = logging.getLogger(__name__)

class RetailVectorStore:
    """Vector store for retail knowledge base retrieval."""
    
    def __init__(self, documents_path="data/retail_documents.txt", persist_directory="./chroma_db"):
        """Initialize the vector store with retail documents."""
        self.documents_path = documents_path
        self.persist_directory = persist_directory
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        
        # Initialize embeddings
        self._initialize_embeddings()
        
        # Load or create vector store
        self._load_or_create_vectorstore()
        
    def _initialize_embeddings(self):
        """Initialize sentence transformers for embeddings."""
        try:
            # Use a lightweight but effective embedding model
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("Embeddings initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise
    
    def _load_or_create_vectorstore(self):
        """Load existing vector store or create new one from documents."""
        try:
            if os.path.exists(self.persist_directory):
                # Load existing vector store
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                logger.info("Loaded existing vector store")
            else:
                # Create new vector store from documents
                self._create_vectorstore_from_documents()
                
            # Initialize retriever
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
        except Exception as e:
            logger.error(f"Failed to load/create vector store: {e}")
            raise
    
    def _create_vectorstore_from_documents(self):
        """Create vector store from retail documents."""
        try:
            # Load documents
            loader = TextLoader(self.documents_path)
            documents = loader.load()
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            splits = text_splitter.split_documents(documents)
            
            # Create vector store
            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            # Persist the vector store
            self.vectorstore.persist()
            logger.info(f"Created new vector store with {len(splits)} document chunks")
            
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise
    
    def search_documents(self, query, k=5):
        """Search for relevant documents based on query."""
        try:
            if not self.retriever:
                raise ValueError("Retriever not initialized")
                
            docs = self.retriever.get_relevant_documents(query)
            return docs[:k]
            
        except Exception as e:
            logger.error(f"Failed to search documents: {e}")
            return []
    
    def get_context_for_query(self, query, k=3):
        """Get relevant context for a query."""
        try:
            docs = self.search_documents(query, k=k)
            context = "\n\n".join([doc.page_content for doc in docs])
            return context
            
        except Exception as e:
            logger.error(f"Failed to get context: {e}")
            return ""
    
    def add_document(self, text, metadata=None):
        """Add a new document to the vector store."""
        try:
            if not self.vectorstore:
                raise ValueError("Vector store not initialized")
                
            # Split the text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_text(text)
            
            # Add to vector store
            self.vectorstore.add_texts(splits, metadatas=[metadata] * len(splits) if metadata else None)
            self.vectorstore.persist()
            
            logger.info(f"Added document with {len(splits)} chunks")
            
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            raise
    
    def similarity_search(self, query, k=5, filter_dict=None):
        """Perform similarity search with optional filtering."""
        try:
            if not self.vectorstore:
                raise ValueError("Vector store not initialized")
                
            results = self.vectorstore.similarity_search(
                query, 
                k=k, 
                filter=filter_dict
            )
            return results
            
        except Exception as e:
            logger.error(f"Failed to perform similarity search: {e}")
            return []
    
    def get_collection_stats(self):
        """Get statistics about the vector store collection."""
        try:
            if not self.vectorstore:
                return {"error": "Vector store not initialized"}
                
            collection = self.vectorstore._collection
            count = collection.count()
            
            return {
                "total_documents": count,
                "embedding_dimension": self.embeddings.client.get_sentence_embedding_dimension(),
                "persist_directory": self.persist_directory
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}

# Global vector store instance
vector_store = None

def initialize_vector_store():
    """Initialize the global vector store instance."""
    global vector_store
    if vector_store is None:
        vector_store = RetailVectorStore()
    return vector_store

def get_vector_store():
    """Get the global vector store instance."""
    global vector_store
    if vector_store is None:
        vector_store = initialize_vector_store()
    return vector_store 