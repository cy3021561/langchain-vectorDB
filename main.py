import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as pi
from langchain.chains import RetrievalQA
from langchain import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

if __name__ == "__main__":
    print("Hello VectorSpace!")
    loader = TextLoader("./mediumblogs/mediumblog1.txt")
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(len(texts))

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    docsearch = pi.from_documents(
        texts, embeddings, index_name="medium-blog-embeddings-index"
    )

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )
    query = "What is a vector space? Give me a 15 words answer for a beginner."
    result = qa({"query": query})
    print(result)
