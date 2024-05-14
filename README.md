# InterWeb Explorer

**InterWeb Explorer** is a streamlined application designed to retrieve, read, and analyze websites and PDF documents. It can answer questions or provide summaries based on the content it processes, supporting both online documents and locally attached PDF files.

## Setup

To get started, you will need an API key from ChatOpenAI.

### Requirements:
- **Document Loaders**: The app uses `UnstructuredURLLoader` and `UnstructuredPDFLoader` to fetch documents.
- **Text Processing**: For more manageable text analysis, the app includes functionality to split large text blocks.
- **Vectorstore and Embeddings**: Utilize `Chroma` and `OpenAIEmbeddings` for vector storage and embedding tasks.
- **Retrieval**: Post-vectorization, the app employs a retrieval system to fetch relevant document sections.
- **Language Models**: The app relies on `ChatOpenAI` for generating responses and summaries.

### Configuration:
Set the required environment variables or use `st.secrets` in your `.streamlit/secrets.toml` file. If you prefer to set environment variables directly, here's how you can do it:

```python
import os
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

## Run
```
streamlit run Interweb_ExploreQ&A.py
```

