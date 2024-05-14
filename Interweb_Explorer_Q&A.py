import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
from unstructured.partition.html import partition_html
from langchain.document_loaders import UnstructuredURLLoader
import wget
import requests
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Load environment variables
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)
openai_api_key = os.environ['openai_api_key']

def download_pdf(url, local_path):

    try:
        # Use wget to download the file directly
        wget.download(url, local_path)
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False



def check_and_explain_url_errors(url):
    if not url:
        return "Error: No URL provided. Please enter a URL."

    try:
        # Send a request to the URL
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        
        # Check for HTTP errors and explain them
        if response.status_code == 400:
            return "400 Bad Request: The server could not understand the request due to invalid syntax."
        elif response.status_code == 401:
            return "401 Unauthorized: Authentication is required and has failed or has not yet been provided."
        elif response.status_code == 403:
            return "403 Forbidden: The server understood the request, but is refusing authorization."
        elif response.status_code == 404:
            return "404 Not Found: The server can't find the requested URL."
        elif response.status_code == 500:
            return "500 Internal Server Error: The server encountered an unexpected condition that prevented it from fulfilling the request."
        elif response.status_code == 502:
            return "502 Bad Gateway: The server, while acting as a gateway or proxy, received an invalid response from the upstream server."
        elif response.status_code == 503:
            return "503 Service Unavailable: The server is not ready to handle the request, possibly due to maintenance or overload."
        elif response.status_code == 504:
            return "504 Gateway Timeout: The server was acting as a gateway or proxy and did not receive a timely response from the upstream server."
        elif response.status_code == 200:
             return "200 OK: The request has succeeded."
        else:
            return f"Received a HTTP status code: {response.status_code}."


    except requests.exceptions.ConnectionError:
        return "Connection Error: Failed to connect to the URL. Please check your connection or the URL."
    except requests.exceptions.Timeout:
        return "Timeout Error: The request to the URL timed out. Please try again later."
    except requests.exceptions.RequestException as e:
        return f"Request Exception: An unexpected error occurred: {e}"





# Set up the Streamlit page
st.set_page_config(page_title="Web Insight AI", page_icon="🌐")
st.sidebar.image("img/cat1.jpeg")
st.header("Interweb Explorer")
st.info("I am an AI capable of answering questions by exploring, reading, and summarizing web content. Additionally, I can interpret and summarize PDF documents found on websites.")


# Initialize session state for the counter and previous URL
if 'ct' not in st.session_state:
    st.session_state.ct = 0
if 'prev_url' not in st.session_state:
    st.session_state.prev_url = ""
if 'prev_pdf' not in st.session_state:
    st.session_state.prev_pdf = None
if 'loader' not in st.session_state:
    st.session_state.loader = None   
# Initialize session state
if 'latest_source' not in st.session_state:
    st.session_state.latest_source = None     


# User inputs
put_url = st.text_input("Insert the link:")


# if put_url:
#     error_message = check_and_explain_url_errors(put_url)
#     st.error(error_message)
# else:
#     st.warning("Please enter a URL to begin.")

     

uploaded_file = st.file_uploader("Choose a file", type=['pdf'])

def save_uploadedfile(uploadedfile):
    # Ensure the 'tempDir' directory exists
    if not os.path.exists('tempDir'):
        os.makedirs('tempDir')
    file_path = os.path.join("tempDir", uploadedfile.name)    
    with open(file_path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success(f"Saved file: {uploadedfile.name} to tempDir")

# Check if a file is uploaded
if uploaded_file is not None:
    # Check if the current file is different from the previously uploaded file
    if uploaded_file != st.session_state.prev_pdf:
        st.session_state.prev_pdf = save_uploadedfile(uploaded_file)
        # Clear existing loader and docs
        st.session_state.loader = None
        st.session_state.docs = None
        # Use the full path for the loader
        full_path = os.path.join("tempDir", uploaded_file.name)
        st.session_state.loader = UnstructuredPDFLoader(full_path) 
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.latest_source = 'pdf_attach_state'

if put_url:
    error_message = check_and_explain_url_errors(put_url)
    st.error(error_message)
    if put_url != st.session_state.prev_url:
        if put_url.endswith('.pdf'):
            # Specify the local path where the PDF should be saved
            download_successful = download_pdf(put_url, "tempDir/pdf_url.pdf")
            if download_successful:
                st.success("PDF downloaded successfully.")
                # Clear existing loader and docs
                st.session_state.loader = None
                st.session_state.docs = None
                #loader = UnstructuredPDFLoader("pdf_url.pdf")
                st.session_state.loader = UnstructuredPDFLoader("tempDir/pdf_url.pdf")
                st.session_state.docs = st.session_state.loader.load()
                st.session_state.latest_source = 'pdf_url_state'
                
            else:
                st.error("Failed to download PDF.")
        else:
            # Process the URL with partition_html if it's not a PDF
            try:
                elements = partition_html(url=put_url)
            except ValueError as e:
                if "403" in str(e):
                    st.warning("Error 403 Forbidden: Access to the URL was denied. This could be due to insufficient permissions, User-Agent restrictions, or other access controls set by the server. Please ensure that your request complies with the server's requirements.")
                else:
                    st.warning(f"An unexpected error occurred: {e}")

            try:
                # Process and filter links
                links = {element.metadata.link_urls[0] for element in elements if element.metadata.link_urls}
                pdf_links = {link for link in links if link.endswith('.pdf')}
                https_links = {link for link in links if link.startswith("https")}
                # Clear existing loader and docs
                st.session_state.loader = None
                st.session_state.docs = None
                #loader = UnstructuredURLLoader(urls=https_links, show_progress_bar=True)
                st.session_state.loader = UnstructuredURLLoader(urls=https_links, show_progress_bar=True)
                st.session_state.docs = st.session_state.loader.load()
                st.session_state.latest_source = 'url_state'
                # Proceed with using 'elements' for further processing
                st.write(f"""HTML partitioned successfully and question will be answered
                    from {len(https_links)} url links.""")
                st.write(f"{len(pdf_links)} PDF links found, not included in the answer.")
            except Exception as e:
                st.error(f"Failed to process HTML: {e}")
else:
    st.warning("Please enter a URL to begin.")                
##################################

# Answer generation setup
question = st.text_input("Ask a question:")
if st.button('Submit'):
    if (put_url and question and st.session_state.loader) or (uploaded_file and question and st.session_state.loader):

        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        all_splits = text_splitter.split_documents(st.session_state.docs)
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings(openai_api_key=openai_api_key))
        retriever = vectorstore.as_retriever(search_type="similarity")
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)
        prompt = ChatPromptTemplate.from_messages(
        ("human", """
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.\n
        If you don't know the answer, just say that you don't know. 
        Question: {question} 
        Context: {context} 
        Answer:
        """)      
        )
        rag_chain = ({"context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)) , "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
        answer = ""
        for chunk in rag_chain.stream(question):
            answer += chunk
        st.write(f"Answer: {answer}")
        st.session_state.prev_url = put_url
        st.session_state.prev_pdf = uploaded_file
        st.session_state.ct += 1
        st.write(f"Number of Questions: {st.session_state.ct}")
        if st.session_state.latest_source == 'pdf_url_state':
            st.success("PDF content loaded and ready for processing.")
        elif st.session_state.latest_source == 'url_state':
            st.success("URL content loaded and ready for processing.")
        elif st.session_state.latest_source == 'pdf_attach_state':
            st.success("from PDF attach")
            st.write(f"upload: , {uploaded_file}")
            st.write(f"prev_pdf, {st.session_state.prev_pdf}")

if os.path.exists("tempDir/pdf_url.pdf"): 
    os.remove("tempDir/pdf_url.pdf")



# import streamlit as st
# import os

# if 'prev_pdf' not in st.session_state:
#     st.session_state.prev_pdf = None

# def save_uploadedfile(uploadedfile):
#     with open(os.path.join("tempDir", uploadedfile.name), "wb") as f:
#         f.write(uploadedfile.getbuffer())
#     return st.success("Saved file:{} to tempDir".format(uploadedfile.name))

# st.title('PDF File Upload')

# uploaded_file = st.file_uploader("Choose a file", type=['pdf'])
# if uploaded_file is not None:
#     if uploaded_file != st.session_state.prev_pdf:
#     # To read file as bytes:
#         bytes_data = uploaded_file.getvalue()
#         st.write("Filename:", uploaded_file.name)
#         # To save file to disk
#         save_uploadedfile(uploaded_file)
#         # To display the PDF file
#         st.write("PDF File Content:")
#         st.write(f"upload: , {uploaded_file}")
#         st.write(f"prev_pdf, {st.session_state.prev_pdf}")
#         st.download_button(label="Download PDF", data=bytes_data, file_name=uploaded_file.name, mime='application/octet-stream')



           
