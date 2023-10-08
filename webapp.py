import streamlit as st
import tempfile

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import pinecone
from langchain.vectorstores import Pinecone, Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

from langchain.prompts import PromptTemplate
from langchain.llms import Clarifai
from langchain.chains import RetrievalQA

st.set_page_config('StarScanAI', ':star:')

@st.cache_resource
def load_models():
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    llm = Clarifai(pat=st.secrets['ClarifaiToken'], user_id='openai', 
                   app_id='chat-completion', model_id='GPT-4')

    pinecone.init(
        api_key=st.secrets['pinecone_apikey'],
        environment=st.secrets['pinecone_env']
    )

    index = pinecone.Index('nasa-rag')
    vectorstore = Pinecone(index, embedding_function, 'text')

    return embedding_function, llm, vectorstore

qna_template = '''
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer in English
'''

report_template = '''
You are STAR, an AI-powered app designed for NASA. Your goal is to
provide structured recommendations for technical requirements.
 The recommendations should include:
 - Section number and version letter of the document under review.
 - Current language from that section.
 - Identified issues.
 - Suggested modifications.
 - Sources of each recommendation.

Use the following pieces of context to provide your answer in the structured format above. 
{context}
'''

qna_prompt = PromptTemplate(
    template=qna_template, input_variables=["context", "question"]
)

report_prompt = PromptTemplate(
    template=report_template, input_variables=["context"]
)

embedding_function, llm, vectorstore = load_models()

st.title('NASA Space Apps: :star: StarScanAI')
tabs = st.tabs(['QnA about NASA Tech', 'Analyze NASA Document'])

with tabs[0]:
    st.write('The Agent will strictly provide information only if related to [NASA Technical Bulletins](https://www.nasa.gov/nesc/knowledge-products/nesc-technical-bulletins/)')
    st.write('It will provide the document URL along with an answer to the query.')
    question = st.text_input('What do you want to know about NASA Technology?')

    if st.button('Ask'):
        qna_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", 
                    retriever=vectorstore.as_retriever(), 
                    chain_type_kwargs={"prompt": qna_prompt})
        with st.spinner('Processing Answer..'):
            answer = qna_chain.run(question)
            result = vectorstore.similarity_search_with_score(question, k=1)
        st.write(answer)
        if result[0][1] > 0.5:
            st.write(f'Source: {result[0][0].metadata["source"]}')

with tabs[1]:
    st.write('Receive recommendations about NASA technical requirements from documents you upload! ')
    pdf_file = st.file_uploader('Upload a NASA PDF Document!', type=['pdf'])
    mode = st.radio('Select an option:', ['Provide recommendations', 'Ask a question'])
    if mode == 'Ask a question':
        question = st.text_input('Ask a question about this Document!')
    
    if pdf_file:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(pdf_file.read())

        pdf_loader = PyPDFLoader(tfile.name)
        pdf_data = pdf_loader.load()

        text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
        docs = text_splitter.split_documents(pdf_data)
        
        pdf_vectors = Chroma.from_documents(documents=docs, embedding=embedding_function)

        if mode == 'Provide recommendations':
            chain_type_kwargs={"prompt": report_prompt}
            prompt = 'none'
        else:
            chain_type_kwargs={"prompt": qna_prompt}
            prompt = question

        report_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", 
                                        retriever=pdf_vectors.as_retriever(), 
                                        chain_type_kwargs=chain_type_kwargs)
        
        if st.button('Submit'):
            with st.spinner('Processing..'):
                report = report_chain.run(prompt)
            st.write(report)
