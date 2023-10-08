# nasa-space-apps

**IMPORTANT**: To run app locally, create `.streamlit/secrets.toml`, and place your ClarifaiToken, pinecone_apikey, pinecone_env there

## QnA about NASA Tech
Check out `RAG_Data-Collection.ipynb` notebook for data collection process

- Retrieve all PDF Links from [NASA Tech Bulletins](https://www.nasa.gov/nesc/knowledge-products/nesc-technical-bulletins/).
- Generate Documents by downloading PDFs and split content into chunks with time interval between requests.
- Use open-source hugging-face embedding model `sentence-transformers/all-MiniLM-L6-v2` to embed documents.
- Initialize Pinecone Online Vector DB and upload embeddings with source metadata.
- Deploy in `RetrievalQA` chain with GPT-4 LLM accessed from Clarifai API.

## Analyze NASA Document
- Load uploaded PDF file
- Split document into chunks
- Generate embeddings
- Store in ChromaDB
- Wrap in `RetrievalQA` chain with prompt engineering
