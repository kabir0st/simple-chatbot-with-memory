import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from tqdm import tqdm

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")


def refresh_vectorize_books(books, embeddings, collection_name="books_vector"):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n::\n", "\n\n", "\n", " ", ""])

    raw_knowledge = []

    for doc in tqdm(books, desc='Processing books'):
        content = f"""
        Title: {doc['name']}
        Description: {doc['description']}
        Authors: {', '.join(doc['authors'])}
        Genres: {', '.join(doc['genres'])}
        """
        raw_knowledge.append(
            Document(page_content=content.strip(),
                     metadata={
                         'source': doc['id'],
                         'title': doc['name'],
                         'authors': ', '.join(doc['authors']),
                         'genres': ', '.join(doc['genres']),
                         'barcode': doc['barcode']
                     }))

    processed_knowledge = text_splitter.split_documents(raw_knowledge)
    print('Storing vectors in Qdrant . . .')
    Qdrant.from_documents(processed_knowledge,
                          embeddings,
                          url=QDRANT_URL,
                          collection_name=collection_name,
                          force_recreate=True)
    print(f"Successfully stored {len(processed_knowledge)} vectors in Qdrant")


def get_qdrant_retriever(client, embeddings):
    return Qdrant(client=client,
                  collection_name="books_vector",
                  embeddings=embeddings).as_retriever(search_kwargs={"k": 30})
