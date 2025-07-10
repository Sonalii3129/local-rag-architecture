import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

# Step 1: Set up Ollama model
Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)

# Step 2: Set up embedding model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
Settings.embed_model = embed_model

# Step 3: Read documents
documents = SimpleDirectoryReader("./data/").load_data()

# Step 4: Set up ChromaDB vector store
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("mgs636test")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Step 5: Create the index
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)

# Step 6: Query
query_engine = index.as_query_engine(llm=Settings.llm)
response = query_engine.query("What topics are covered in this class?")
print(response)
