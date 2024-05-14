import re
import os
from pathlib import Path
from dotenv import load_dotenv
from unstructured.partition.html import partition_html
from langchain.document_loaders import UnstructuredURLLoader

# Verify envirnment
import sys
print(sys.executable)
print(sys.version)

# Load environment variables
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)
openai_api_key=os.environ['openai_api_key']

put_url = "https://electrovaya.com/"


elements = partition_html(url=put_url)
print(elements)

links = []
pdf_links = []
for element in elements:
    if element.metadata.link_urls:
        relative_link = element.metadata.link_urls[0]
        print(relative_link)
        if relative_link.endswith('.pdf'):
            pdf_links.append(f"{relative_link}") 
        elif relative_link.startswith("https"):
            print(relative_link)
            links.append(f"{relative_link}")


# Convert list to a set to remove duplicates
unique_set = set(links)
# Convert set back to a list
unique_list = list(unique_set)            


# Loop through the list and check each item
index_of_target = -1
for index, item in enumerate(unique_list):
    if item.startswith(put_url):
        index_of_target = index
        break  # Stop searching once the first match is found
if index_of_target == -1:
    print(f"partition_html does not return '{put_url}'")

# Load documents from the links
loader = UnstructuredURLLoader(urls=unique_list, show_progress_bar=True)
docs = loader.load()


# Assuming docs is a list and we're looking at the first document
doc = docs[2]
page_content = doc.page_content

# Clean up excessive whitespace:
# This regex will replace any sequence of whitespace characters (spaces, tabs, newlines) with a single space,
# except it will preserve paragraph breaks by replacing occurrences of multiple newlines with a single newline.
cleaned_content = re.sub(r'\s*\n\s*\n\s*', '\n\n', page_content)  # Handle paragraph breaks more cleanly
cleaned_content = re.sub(r'[ \t]+', ' ', cleaned_content)          # Replace spaces and tabs in a line with a single space
# Print the cleaned content
print(cleaned_content)




from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

len(all_splits)
len(all_splits[0].page_content)
all_splits[3].metadata



from langchain.chat_models import ChatOpenAI
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


# We can embed and store all of our document splits in a single command using the Chroma vector store and OpenAIEmbeddings model.
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings(openai_api_key=openai_api_key))

# VectorStoreRetrievervuses the similarity search capabilities of a vector store to facilitate retrieval. 
# Any VectorStore can easily be turned into a Retriever with VectorStore.as_retriever():


retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
retrieved_docs = retriever.invoke("what is Electrovaya")
len(retrieved_docs)
print(retrieved_docs[0].page_content)



# Configure the language model for answering questions
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)


# Create prompt
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    ("human", """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.\n
    If you don't know the answer, just say that you don't know. 
    Question: {question} 
    Context: {context} 
    Answer:
    """)
)


# Setup a chain for generating answers
example_messages = prompt.invoke(
    {"context": "filler context", "question": "filler question"}
).to_messages()
example_messages
print(example_messages[1].content)


from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

for chunk in rag_chain.stream("what is the proven execution"):
    print(chunk, end="", flush=True)

##################################
########################

from unstructured.partition.html import partition_html
import requests

put_url = "https://electrovaya.com/press/electrovaya-announces-results-of-annual-meeting-of-shareholders/"  # Replace with the actual URL you want to check

def find_pdf_links(put_url):
    # Fetch the HTML content from the URL
    response = requests.get(put_url)
    if response.status_code == 200:
        # Use unstructured to partition the HTML content
        elements = partition_html(url=put_url)

        # Initialize a list to hold any found PDF links
        pdf_links = []

        # Iterate through elements and search for links
        for element in elements:
            if element.metadata.link_urls:
                relative_link = element.metadata.link_urls[0]
                if relative_link.endswith('.pdf'):
                    print('yes')
                    pdf_links.append(f"{relative_link}")                

        # Return the list of PDF links
        return pdf_links
    else:
        print("Failed to fetch the HTML content")
        return []

put_url = 'https://electrovaya.com/wp-content/uploads/2024/03/ELVA-General-Corporate-Presentation-Website-v1_02-28-2024.pptx.pdf'
h=elements[0].metadata.to_dict()
elements[0].metadata.link_urls[0]
h['url']
# Example usage

put_url = "https://electrovaya.com"
put_url = "https://electrovaya.com/press/electrovaya-announces-results-of-annual-meeting-of-shareholders/"  # Replace with the actual URL you want to check
s = find_pdf_links(put_url)

import wget

def download_pdf(url, local_path):

    try:
        # Use wget to download the file directly
        wget.download(url, local_path)
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


download_pdf(pdf_links[1], "pdf_url2.pdf")
download_pdf(put_url, "pdf_url.pdf")

from langchain_community.document_loaders import UnstructuredPDFLoader
# Load documents from the links

loader = UnstructuredPDFLoader("pdf_url.pdf")
docs = loader.load()




import requests

def download_pd(url, local_path):
    """
    Downloads a PDF from a given URL and saves it to a specified local path.

    Parameters:
        url (str): The URL to download the PDF from.
        local_path (str): The file path where the PDF should be saved.

    Returns:
        bool: True if the download and save were successful, False otherwise.
    """
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        # Check if the request was successful
        if response.status_code == 200:
            # Open the file and write the content in binary mode
            with open(local_path, 'wb') as f:
                f.write(response.content)
            return True
        else:
            print(f"Failed to download. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return False
