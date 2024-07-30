import os

from urllib.request import urlretrieve
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, RobertaTokenizer, AutoConfig
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings


class Chatbot:
    def __init__(self):
        self.persist_directory = None
        self.embedding = None
        self.local_llm = None

    def setup_model(self):
        # Download documents from U.S. Census Bureau to local directory.
        os.makedirs("us_census", exist_ok=True)
        files = [
            "https://www.census.gov/content/dam/Census/library/publications/2022/demo/p70-178.pdf",
            # "https://www.census.gov/content/dam/Census/library/publications/2023/acs/acsbr-017.pdf",
            # "https://www.census.gov/content/dam/Census/library/publications/2023/acs/acsbr-016.pdf",
            # "https://www.census.gov/content/dam/Census/library/publications/2023/acs/acsbr-015.pdf",
        ]
        for url in files:
            file_path = os.path.join("us_census", url.rpartition("/")[2])
            urlretrieve(url, file_path)
        # Files is a local directory

        embedding_model_name = "hkunlp/instructor-large"
        # Pass the directory path where the model is stored on your system
        model_name = r"D:\Navya\Coding\PythonNew\flan-t5-small_"
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Initialize a model for sequence-to-sequence tasks using the specified pretrained model
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Create a pipeline for text-to-text generation using a specified model and tokenizer
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            temperature=0.3,
            top_p=0.95,
            repetition_penalty=1.15
        )

        self.local_llm = HuggingFacePipeline(pipeline=pipe)

        # loading documents

        loader = DirectoryLoader('./us_census', glob="./*.pdf", loader_cls=PyPDFLoader)

        documents = loader.load()

        print(len(documents))

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                       chunk_overlap=200
                                                       )

        texts = text_splitter.split_documents(documents)


        #create embeddings
        instructor_embeddings = HuggingFaceInstructEmbeddings(
            model_name=r"D:\Navya\Coding\PythonNew\instructor-large",
            model_kwargs={"device": "cpu"} , # Specify the device to be used for inference (GPU - "cuda" or CPU - "cpu")
            encode_kwargs = {'normalize_embeddings': True}
        )


        # Define the directory where the embeddings will be stored on disk

        self.persist_directory = 'db'

        # Assign the embedding model (instructor_embeddings) to the 'embedding' variable

        self.embedding = instructor_embeddings
        # Create a Chroma instance and generate embeddings from the supplied texts
        # Store the embeddings in the specified 'persist_directory' (on disk)
        # vectordb = Chroma.from_documents(documents=texts, embedding=self.embedding, persist_directory=self.persist_directory)
        #
        # # Persist the database (vectordb) to disk
        # vectordb.persist()

    def qa_chain(self):
        # Set the vectordb variable to None to release the memory
        vectordb = None
        # Create a new Chroma instance by loading the persisted database from the
        # specified directory and using the provided embedding function.
        vectordb = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding)
        # Create a retriever from the Chroma database (vectordb) with search parameters
        # The value of "k" determines the number of nearest neighbors to retrieve.
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})

        qa_chain = RetrievalQA.from_chain_type(llm=self.local_llm, chain_type="stuff", retriever=retriever,
                                               return_source_documents=True)
        # Example query for the QA chain
        return qa_chain

def get_response(query, qa_chain = None):
    obj = Chatbot()
    obj.setup_model()
    qc = obj.qa_chain()
    res = qc.invoke(query)
    return res["result"]

# while True:
#     query = input("please enter your question")
#     if query == "exit":
#         break
#     funct(query)
#     print(llm_response["result"])


# a = get_response("What were the trends in median household income across different states in the United States between 2021 and 2022.")
# print(a)