import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.prompts import PromptTemplate


# Step 1: Set up Ollama model
Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)

# Step 2: Set up embedding model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
Settings.embed_model = embed_model

# Step 3: Read documents


# Step 4: Set up ChromaDB vector store
from chromadb import PersistentClient

chroma_client = PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("mgs636test")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Step 5: Create the index
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)


# Step 6: Query
custom_prompt = PromptTemplate(
    "You are a helpful AI assistant. Read the following context and answer the user's question clearly.\n\nContext:\n{context_str}\n\nQuestion:\n{query_str}\n\nAnswer:"
)
query_engine = index.as_query_engine(
    llm=Settings.llm,
    text_qa_template=custom_prompt,
    similarity_top_k=2  # <-- this goes here now
)

response = query_engine.query("What topics are covered in this class?")

print(response)
