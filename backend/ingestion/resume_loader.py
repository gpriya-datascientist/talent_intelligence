"""
resume_loader.py — loads and cleans resume text from PDF or raw string.
Uses LangChain document loaders so it plugs directly into the extraction chain.
"""
import re
from pathlib import Path
from typing import Union
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


def load_resume_from_pdf(file_path: Union[str, Path]) -> str:
    """
    Load a PDF resume and return cleaned plain text.
    LangChain PyPDFLoader splits by page — we join them back.
    """
    loader = PyPDFLoader(str(file_path))
    pages: list[Document] = loader.load()
    raw_text = "\n".join(page.page_content for page in pages)
    return clean_resume_text(raw_text)


def load_resume_from_string(raw_text: str) -> str:
    """For seeded/test data where we already have text."""
    return clean_resume_text(raw_text)


def clean_resume_text(text: str) -> str:
    """
    Normalise resume text before sending to LLM:
    - collapse whitespace
    - remove non-ASCII junk from PDF extraction
    - strip page numbers
    """
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\bPage\s+\d+\b', '', text, flags=re.IGNORECASE)
    text = text.strip()
    return text


def chunk_resume_for_llm(text: str, max_chars: int = 3000) -> list[str]:
    """
    Split long resumes into chunks for LLM context window safety.
    Splits on double newlines (section boundaries) first.
    """
    if len(text) <= max_chars:
        return [text]
    sections = text.split("\n\n")
    chunks, current = [], ""
    for section in sections:
        if len(current) + len(section) < max_chars:
            current += section + "\n\n"
        else:
            if current:
                chunks.append(current.strip())
            current = section + "\n\n"
    if current:
        chunks.append(current.strip())
    return chunks
