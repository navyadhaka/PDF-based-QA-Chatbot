#pip install transformers sentence-transformers langchain torch faiss-cpu numpy
#pip install langchain-community
import os # read or write file system
from urllib.request import urlretrieve #used to download files from a URL and save them locally.
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings       #used to create embeddings using models from the Hugging Face library.
from langchain_community.llms import HuggingFacePipeline               #used to integrate Hugging Face pipelines, such as those for language models, into LangChain.
from langchain_community.document_loaders import PyPDFLoader           #used to load and read PDF documents.
from langchain_community.document_loaders import PyPDFDirectoryLoader  #used to load all PDF documents in a specified directory.
from langchain.text_splitter import RecursiveCharacterTextSplitter     #used to split text into smaller chunks, which is useful for processing large documents.
from langchain_community.vectorstores import FAISS                     # library for efficient similarity search and clustering of dense vectors.
from langchain.chains import RetrievalQA                               #used to create a question-answering chain that retrieves relevant documents and extracts answers from them.
from langchain.prompts import PromptTemplate                           #used to create templates for prompts, which can be used to format input text for language models.
from langchain_huggingface import HuggingFaceEmbeddings


# Download documents from U.S. Census Bureau to local directory.
os.makedirs("us_census", exist_ok=True)
files = [
    "https://www.census.gov/content/dam/Census/library/publications/2022/demo/p70-178.pdf",
    "https://www.census.gov/content/dam/Census/library/publications/2023/acs/acsbr-017.pdf",
    "https://www.census.gov/content/dam/Census/library/publications/2023/acs/acsbr-016.pdf",
    "https://www.census.gov/content/dam/Census/library/publications/2023/acs/acsbr-015.pdf",
]
for url in files:
    file_path = os.path.join("us_census", url.rpartition("/")[2])
    urlretrieve(url, file_path)

# Load pdf files in the local directory
loader = PyPDFDirectoryLoader("./us_census/")

docs_before_split = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 700,
    chunk_overlap  = 50,
)
docs_after_split = text_splitter.split_documents(docs_before_split)

#print(docs_after_split[0])


avg_doc_length = lambda docs: sum([len(doc.page_content) for doc in docs])//len(docs)
avg_char_before_split = avg_doc_length(docs_before_split)
avg_char_after_split = avg_doc_length(docs_after_split)

print(f'Before split, there were {len(docs_before_split)} documents loaded, with average characters equal to {avg_char_before_split}.')
print(f'After split, there were {len(docs_after_split)} documents (chunks), with average characters equal to {avg_char_after_split} (average chunk length).')


huggingface_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)


sample_embedding = np.array(huggingface_embeddings.embed_query(docs_after_split[0].page_content))
print("Sample embedding of a document chunk: ", sample_embedding)
print("Size of the embedding: ", sample_embedding.shape)


vectorstore = FAISS.from_documents(docs_after_split, huggingface_embeddings)

query = """What were the trends in median household income across
           different states in the United States between 2021 and 2022."""
         # Sample question, change to other questions you are interested in.
relevant_documents = vectorstore.similarity_search(query)
print(f'There are {len(relevant_documents)} documents retrieved which are relevant to the query. Display the first one:\n')
print(relevant_documents[0].page_content)


# Use similarity searching algorithm and return 3 most relevant documents.
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

from langchain_huggingface import HuggingFaceEndpoint

hf = HuggingFaceEndpoint(
    huggingfacehub_api_token="your_token_here",  #do not change "your_token_here"
    repo_id="mistralai/Mistral-7B-v0.1",
    temperature=0.1,
    model_kwargs={"max_length": 500})

query = """What were the trends in median household income across different states in the United States between 2021 and 2022."""  # Sample question, change to other questions you are interested in.
hf.invoke(query)


