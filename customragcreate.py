from chromadb import PersistentClient
import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
Settings.embed_model = embed_model

# New folder for custom documents
documents = SimpleDirectoryReader("./custom_data/").load_data()

chroma_client = PersistentClient(path="./custom_chroma")
chroma_collection = chroma_client.get_or_create_collection("customrag")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)

query_engine = index.as_query_engine(llm=Settings.llm)
response = query_engine.query("What technical skills and experiences are shown in these documents?")

print(response)
