import os
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings

def get_index(data, index_name):

    if not os.path.exists("./Embeddings"):
        os.makedirs("./Embeddings")

    Settings.embed_model = OllamaEmbedding(
        model_name="nomic-embed-text"
    )

    Settings.llm = Ollama(model="mistral")

    embedding_path = os.path.join("Embeddings",index_name)

    index = None
    if not os.path.exists(embedding_path):
        print("building index", index_name)
        index = VectorStoreIndex.from_documents(
            data, 
            llm=Settings.llm, 
            embed_model=Settings.embed_model, 
            show_progress=True,
        )
        index.storage_context.persist(persist_dir=embedding_path)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=embedding_path)
        )

    return index



