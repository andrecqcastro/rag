from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import chromadb
from llama_index.core.node_parser import SentenceSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding

def indexer(dir_pest:str, dir_data:str):
    db = chromadb.PersistentClient(path=dir_pest)
    chroma_collection = db.get_or_create_collection("quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    lc_embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    embed_model = LangchainEmbedding(lc_embed_model)

    reader = SimpleDirectoryReader(input_dir=dir_data)
    documents = reader.load_data()


    index = VectorStoreIndex.from_documents(
        documents, vector_store=vector_store, storage_context=storage_context,transformations=[SentenceSplitter(chunk_size=1024, chunk_overlap=20)], embed_model=embed_model
    )

indexer('./vector_store/mole cricket', './dados/mole cricket')

