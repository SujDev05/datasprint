from langchain.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from llm.vector_store import get_vector_store
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetailLLMChat:
    def __init__(self, model_name="mistral"):
        """Initialize the LLM chat system with memory, chains, and vector store."""
        try:
            # Initialize Ollama LLM with streaming
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]) #callback manager for streaming the output
            self.llm = Ollama(
                model=model_name,
                callback_manager=callback_manager,
                temperature=0.7
            )
            
            # Initialize conversation memory
            self.memory = ConversationBufferMemory( #memory for the conversation
                memory_key="history",
                return_messages=True
            )
            
            # Create conversation chain
            self.conversation_chain = ConversationChain(
                llm=self.llm,
                memory=self.memory,
                verbose=True
            )
            
            # Initialize vector store
            self.vector_store = get_vector_store()
            
            # Create retrieval QA chain
            self.retrieval_chain = RetrievalQA.from_chain_type( 
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.retriever,
                return_source_documents=True,
                verbose=True
            )
            
            logger.info(f"LLM chat initialized with model: {model_name} and vector store")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def get_response(self, prompt, context=None, use_knowledge_base=False):
        """Get response from LLM with optional knowledge base retrieval."""
        try:
            if use_knowledge_base:
                # Use retrieval QA chain for knowledge base queries
                result = self.retrieval_chain({"query": prompt})
                response = result.get("result", "No relevant information found.")
                source_docs = result.get("source_documents", [])
                
                # Add source information if available
                if source_docs:
                    sources = [doc.metadata.get('source', 'Unknown') for doc in source_docs]
                    response += f"\n\nSources: {', '.join(set(sources))}"
                
                return response
                
            elif context:
                # Enhanced prompt with context
                enhanced_prompt = f"""
                Context: {context}
                
                User Question: {prompt}
                
                Please provide a helpful response based on the context and your knowledge.
                """
                response = self.llm(enhanced_prompt)
            else:
                # Use conversation chain for general chat
                response = self.conversation_chain.predict(input=prompt)
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error getting LLM response: {e}")
            return f"I apologize, but I encountered an error: {str(e)}"
    
    def get_knowledge_base_response(self, query):
        """Get response specifically from the retail knowledge base."""
        try:
            # Get relevant context from vector store
            context = self.vector_store.get_context_for_query(query, k=3)
            
            if context:
                prompt = f"""
                Based on the following retail knowledge base information:
                
                {context}
                
                Please answer this question: {query}
                
                Provide a comprehensive answer using the knowledge base information and your expertise.
                """
                return self.llm(prompt)
            else:
                return "I couldn't find relevant information in the knowledge base for your query."
                
        except Exception as e:
            logger.error(f"Error getting knowledge base response: {e}")
            return f"I encountered an error while searching the knowledge base: {str(e)}"
    
    def search_knowledge_base(self, query, k=5):
        """Search the knowledge base for relevant documents."""
        try:
            docs = self.vector_store.search_documents(query, k=k)
            return docs
        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            return []
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear()
        logger.info("Conversation memory cleared")
    
    def get_memory_summary(self):
        """Get a summary of the conversation history."""
        return self.memory.buffer
    
    def get_vector_store_stats(self):
        """Get statistics about the vector store."""
        try:
            return self.vector_store.get_collection_stats()
        except Exception as e:
            logger.error(f"Error getting vector store stats: {e}")
            return {"error": str(e)}

# Initialize global chat instance
chat_instance = RetailLLMChat()

def get_llm_response(prompt, context=None, use_knowledge_base=False):
    """Wrapper function for backward compatibility."""
    return chat_instance.get_response(prompt, context, use_knowledge_base)

def get_knowledge_base_response(query):
    """Get response from retail knowledge base."""
    return chat_instance.get_knowledge_base_response(query)

def search_knowledge_base(query, k=5):
    """Search the knowledge base."""
    return chat_instance.search_knowledge_base(query, k)

def clear_chat_memory():
    """Clear the chat memory."""
    chat_instance.clear_memory()

def get_chat_history():
    """Get the current chat history."""
    return chat_instance.get_memory_summary()

def get_vector_store_stats():
    """Get vector store statistics."""
    return chat_instance.get_vector_store_stats() 