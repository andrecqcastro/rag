from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import chromadb

def retriever(dir_pest:str, question:str):
    '''Descrição função'''
    db2 = chromadb.PersistentClient(path=dir_pest)
    chroma_collection = db2.get_or_create_collection("quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    lc_embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    embed_model = LangchainEmbedding(lc_embed_model)

    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context, embed_model=embed_model)

    nodes = index.as_retriever(similarity_top_k=1).retrieve(question)
    results = [node.text for node in nodes]
    context = results[0]

    return context


def llm(context:str, pest:str, question:str):
    '''Descrição da função'''
    
    prompt =  f"""Você deverá responder a dúvidas relacionadas a uma praga agrícola.

    Contexto da pergunta: {context}.
    
    Praga cuja pergunta se refere: {pest}.
    
    Pergunta: {question}"""

    model = Ollama(model='phi3:mini')

    answer = model.invoke(prompt)
    print(prompt)
    print(answer)
    return answer

context = retriever('./vector_store/aphids', 'Pulgão causa muito dano à safra?')

llm(str(context), 'Pulgões', 'Pulgão causa muito dano à safra?')