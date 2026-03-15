"""
ingest_laws.py
==============
Builds a persistent FAISS index from Arabic legal documents.

This script:
- Reads Markdown (.md) and PDF (.pdf) files
- Preserves legal structure for Markdown (sections / articles)
- Extracts text safely from PDFs
- Builds embeddings using a top-tier Arabic-capable model
- Saves the FAISS index to disk for reuse after app restarts

"""

import os
import json

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader


# ======================================================
# Configuration
# ======================================================

LAWS_FOLDER = "cleaned_data"               # Folder containing .md and .pdf files
FAISS_INDEX_FOLDER = "faiss_laws_index" # Persistent FAISS index location

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
# EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"


def make_legal_splitter() -> RecursiveCharacterTextSplitter:
    """Custom splitter respecting Arabic legal document structure."""
    return RecursiveCharacterTextSplitter(
        separators=[
            "\nالمادة ",  # Article boundary
            "\nالبند ",   # Clause boundary
            "\nالفصل ",  # Chapter boundary
            "\n\n",      # Paragraph break
            ". ",        # Sentence boundary
            "\n"         # Line break
        ],
        chunk_size=1000,
        chunk_overlap=200,
    )


def load_jsonl_chunks():
    """
    Load pre-chunked JSONL documents.

    Each line must be a JSON object that includes a `content` field (string),
    plus any optional metadata fields. This is the recommended format for
    high-quality RAG ingestion (hierarchical chunks + alt-text for images).

    Supported keys:
      - content: required (string)
      - any other keys are stored as metadata
    """
    documents = []

    for filename in os.listdir(LAWS_FOLDER):
        if not filename.lower().endswith(".jsonl"):
            continue

        path = os.path.join(LAWS_FOLDER, filename)

        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception as e:
                    raise ValueError(f"Invalid JSONL in {filename} at line {line_no}: {e}")

                content = obj.get("content")
                if not content or not isinstance(content, str):
                    continue

                meta = {k: v for k, v in obj.items() if k != "content"}

                # Normalize/guarantee common metadata keys expected by the web app
                # - filename: shown to the user as a "source"
                if "filename" not in meta:
                    meta["filename"] = meta.get("source_file", filename)

                # - source: used for routing (laws vs firecode, etc.)
                if "source" not in meta:
                    # prefer `doc_id` (our firecode chunks use doc_id="firecode")
                    meta["source"] = meta.get("doc_id") or meta.get("source") or "unknown"

                meta["format"] = "jsonl_chunk"

                documents.append(Document(page_content=content, metadata=meta))

    return documents

def load_markdown_documents():
    """
    Load Markdown law files as raw documents.

    Markdown is treated as the primary source of rules,
    tables, and enforceable regulations.
    Splitting happens later with make_legal_splitter().
    """
    documents = []

    for filename in os.listdir(LAWS_FOLDER):
        if not filename.lower().endswith(".md"):
            continue

        path = os.path.join(LAWS_FOLDER, filename)

        with open(path, "r", encoding="utf-8") as f:
            markdown_text = f.read()

        doc = Document(
            page_content=markdown_text,
            metadata={
                "filename": filename,
                "format": "markdown",
                "source": "laws",
            }
        )
        documents.append(doc)

    return documents


def load_pdf_documents():
    """
    Load PDF files and extract text page by page.

    PDFs are treated as authoritative references
    (gazettes, official publications, scanned regulations).
    """
    documents = []

    for filename in os.listdir(LAWS_FOLDER):
        if not filename.lower().endswith(".pdf"):
            continue

        path = os.path.join(LAWS_FOLDER, filename)

        loader = PyPDFLoader(path)
        pdf_pages = loader.load()

        for page in pdf_pages:
            page.metadata.update({
                "filename": filename,
                "format": "pdf",
                "source": "laws",
            })

        documents.extend(pdf_pages)

    return documents


def build_and_save_index(documents):
    """
    Split documents into chunks, generate embeddings,
    and save the FAISS index to disk.
    """

    # Custom splitter respecting Arabic legal document structure
    splitter = make_legal_splitter()

    chunks = []
    for d in documents:
        # Pre-chunked JSONL docs should NOT be re-split.
        if isinstance(getattr(d, "metadata", None), dict) and d.metadata.get("format") == "jsonl_chunk":
            chunks.append(d)
        else:
            chunks.extend(splitter.split_documents([d]))

    print(f"\nGenerating embeddings for {len(chunks)} chunks...")
    print("(This may take 15-30 minutes depending on your hardware)")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={'normalize_embeddings': True},
    )

    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    print("Saving index to disk...")
    vectorstore.save_local(FAISS_INDEX_FOLDER)

    print(f"✓ FAISS index successfully saved to '{FAISS_INDEX_FOLDER}'")


# ======================================================
# Entry point
# ======================================================

def main():
    print("Starting legal document ingestion...")

    if not os.path.exists(LAWS_FOLDER):
        raise RuntimeError("Law data folder does not exist")

    jsonl_docs = load_jsonl_chunks()
    markdown_docs = load_markdown_documents()
    pdf_docs = load_pdf_documents()

    all_docs = jsonl_docs + markdown_docs + pdf_docs

    print(f"Loaded {len(jsonl_docs)} JSONL chunks")
    print(f"Loaded {len(markdown_docs)} Markdown sections")
    print(f"Loaded {len(pdf_docs)} PDF pages")
    print(f"Total documents before chunking: {len(all_docs)}")

    build_and_save_index(all_docs)

    print("Ingestion completed successfully")


if __name__ == "__main__":
    main()
