from src.helper import pdf_load,text_split,download_hugging_face_embeddings
import pinecone
from langchain.vectorstores import Pinecone
from dotenv import load_dotenv

import os
load_dotenv()
PINECONE_API_KEY=os.environ.get("PINECONE_API_KEY")
PINECONE_API_ENV=os.environ.get("PINECONE_API_ENV")
print(PINECONE_API_KEY)
print(PINECONE_API_ENV)



extracted_data=pdf_load("pdf_data/")
text_chunk=text_split(extracted_data=extracted_data)
text_chunk=text_split(extracted_data=extracted_data)
embedding = download_hugging_face_embeddings()

#Initializing the Pinecone

pinecone.init(api_key=PINECONE_API_KEY,
              environment=PINECONE_API_ENV)

index_name="test"
#Creating Embeddings for Each of The Text Chunks & storing
#docsearch=Pinecone.from_texts([t.page_content for t in text_chunk], embedding, index_name=index_name)
docsearch=Pinecone.from_existing_index(index_name, embedding)

