from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("qna.csv", comment='#', quotechar='"')
embeddings = OllamaEmbeddings(model="mxbai-embed-large")


db_location = "./chroma_langchain_qna_db"

# Check and see if the db already exists, if not, add documents
add_documents = not os.path.exists(db_location)

# If we need to add documents
# Perform adding documents
if add_documents:
    documents = []
    ids = []

    # Go row by row through csv file to access entries
    for i, row in df.iterrows():
        document = Document(

            # page content is what is being vectorized and what we will be looking up
            # Anything we need to look up in the db needs to go in the page content
            # row["Title"] + " " + row["Review"] is the format of the data
            page_content=row["Question"] + " " + row["Answer"],

            # Metadat is additional information included in docs
            # We won't be querying based on metadata
            metadata={"question": row["Question"], "answer": row["Answer"]},

            # Index of the data by row is the id
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)

# Create and add to the vector store
vector_store = Chroma(
    collection_name = "qna",
    # Store permanently
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(
    # Specify the number of arguments we want to lookup
    # Look up 5 relevant reviews and pass to llm
    search_kwargs={"k": 5}
)