from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

# 1. Load documents
loader = TextLoader(r"C:\Users\Administrator\Desktop\Product_Breakdown.txt", encoding="utf-8")
documents = loader.load()

# 2. Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

# 3. Create embeddings
embeddings = OllamaEmbeddings(model="llama3.2:1b")

# 4. Create vector store
vectorstore = FAISS.from_documents(chunks, embeddings)

# 5. Create retriever
retriever = vectorstore.as_retriever()

# 6. Create QA chain
llm = OllamaLLM(model="llama3.2:1b")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

# 7. Ask question
response = qa_chain.run("you must give me summery of this file")
print(response)