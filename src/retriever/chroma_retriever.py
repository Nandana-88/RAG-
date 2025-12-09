from langchain_community.vectorstores import Chroma

def get_chroma_retriever(persist_directory, embedding, k=5):
    chroma = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )

    return chroma.as_retriever(search_kwargs={"k": k})
