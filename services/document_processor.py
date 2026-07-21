"""
Document Processing Service
Reads the active corpus (registered by ``config.corpus_registry``), chunks
the text, and prepares it for embedding.
"""

import json
import unicodedata
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re

from config import config, get_active_documents

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_folder_name(name: str) -> str:
    """Normalize a folder name so Unicode variant characters fold to ASCII."""
    normalized = unicodedata.normalize("NFD", name)
    ascii_quote = chr(0x22)
    ascii_single = chr(0x27)
    normalized = (
        normalized.replace(chr(0x201C), ascii_quote)
        .replace(chr(0x201D), ascii_quote)
        .replace(chr(0x2018), ascii_single)
        .replace(chr(0x2019), ascii_single)
    )
    return normalized


@dataclass
class TextChunk:
    text: str
    source: str
    page: Optional[str] = None
    work_id: Optional[str] = None
    author: Optional[str] = None
    title: Optional[str] = None
    chunk_index: int = 0


@dataclass
class ProcessedDocument:
    work_id: str
    title: str
    author: str
    year: str
    genre: str
    chunks: List[TextChunk]
    total_pages: int
    total_text_length: int


class DocumentProcessor:
    """Processes the active corpus into structured text chunks for RAG."""

    def __init__(self):
        self.ocr_path = config.paths.processed_output_path
        self.output_path = config.paths.get_absolute_path(config.paths.processed_texts_path)
        self.chunk_size = config.rag.chunk_size
        self.chunk_overlap = config.rag.chunk_overlap

        self.output_path.mkdir(parents=True, exist_ok=True)

    def _documents(self) -> List[dict]:
        """Return the currently active corpus, falling back to demo synthesis."""
        return get_active_documents()

    def process_all_documents(self) -> List[ProcessedDocument]:
        processed_docs: List[ProcessedDocument] = []
        for doc in self._documents():
            try:
                processed = self.process_single_document(doc.get("folder", doc.get("id")))
                if processed:
                    processed_docs.append(processed)
                    logger.info(f"Successfully processed: {doc.get('title') or doc.get('id')}")
            except Exception as e:
                logger.error(f"Error processing {doc.get('folder', doc.get('id'))}: {str(e)}")
                continue
        logger.info(f"Processed {len(processed_docs)} documents successfully")
        return processed_docs

    def process_single_document(self, folder_name: str) -> Optional[ProcessedDocument]:
        """Process a single document given the on-disk folder name."""
        work_metadata = next(
            (d for d in self._documents() if d.get("folder") == folder_name),
            None,
        )
        if not work_metadata:
            logger.warning(f"No metadata found for folder: {folder_name}")
            return None

        ocr_filename = work_metadata.get("ocr_file", "ocr_output.json")
        ocr_file_path = self.ocr_path / folder_name / ocr_filename

        if not ocr_file_path.exists():
            # Fuzzy match: folders on disk may use slightly different Unicode
            # characters than the configured folder name.
            normalized_config = normalize_folder_name(folder_name)
            if self.ocr_path.exists():
                for actual_folder in self.ocr_path.iterdir():
                    if actual_folder.is_dir():
                        if normalize_folder_name(actual_folder.name) == normalized_config:
                            ocr_file_path = actual_folder / ocr_filename
                            logger.info(
                                f"Found matching folder: '{actual_folder.name}' for config: '{folder_name}'"
                            )
                            break

        if not ocr_file_path.exists():
            logger.warning(f"OCR file not found: {ocr_file_path}")
            return None

        try:
            with open(ocr_file_path, "r", encoding="utf-8") as f:
                ocr_data = json.load(f)

            cleaned_pages = self._extract_and_clean_text(ocr_data)
            chunks = self._create_chunks(cleaned_pages, work_metadata)
            total_text = " ".join([page_text for _, page_text in cleaned_pages])

            processed_doc = ProcessedDocument(
                work_id=work_metadata["id"],
                title=work_metadata.get("title", work_metadata["id"]),
                author=work_metadata.get("author", ""),
                year=work_metadata.get("year", ""),
                genre=work_metadata.get("genre", ""),
                chunks=chunks,
                total_pages=len(cleaned_pages),
                total_text_length=len(total_text),
            )

            self._save_processed_document(processed_doc)
            return processed_doc

        except Exception as e:
            logger.error(f"Error processing document {folder_name}: {str(e)}")
            return None

    def _extract_and_clean_text(self, ocr_data: Dict) -> List[Tuple[str, str]]:
        cleaned_pages = []
        for page_key, page_text in ocr_data.items():
            if not isinstance(page_text, str):
                continue
            cleaned_text = self._clean_text(page_text)
            if len(cleaned_text.strip()) < 50:
                continue
            if self._is_metadata_page(cleaned_text):
                continue
            cleaned_pages.append((page_key, cleaned_text))
        return cleaned_pages

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"^\d+→\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"[→←↑↓]", "", text)
        text = text.replace("。", ". ")
        text = text.replace("、", ", ")
        text = re.sub(r"\.{3,}", "...", text)
        text = re.sub(r"-{3,}", "---", text)
        text = re.sub(r"\s+([.!?])", r"\1", text)
        text = re.sub(r"([.!?])\s+", r"\1 ", text)
        return text.strip()

    def _is_metadata_page(self, text: str) -> bool:
        metadata_indicators = [
            "copyright",
            "all rights reserved",
            "published by",
            "isbn",
            "contents",
            "table of contents",
            "acknowledgments",
            "bibliography",
            "index",
        ]
        text_lower = text.lower()
        for indicator in metadata_indicators:
            if indicator in text_lower:
                return True
        lines = text.split("\n")
        short_lines = sum(1 for line in lines if len(line.strip()) < 10)
        if lines and short_lines / len(lines) > 0.7:
            return True
        return False

    def _create_chunks(self, pages: List[Tuple[str, str]], work_metadata: Dict) -> List[TextChunk]:
        chunks: List[TextChunk] = []
        chunk_index = 0
        for page_key, page_text in pages:
            page_chunks = self._split_text_into_chunks(page_text)
            for chunk_text in page_chunks:
                chunks.append(TextChunk(
                    text=chunk_text,
                    source=f"{work_metadata['id']}-{page_key}",
                    page=page_key,
                    work_id=work_metadata["id"],
                    author=work_metadata.get("author", ""),
                    title=work_metadata.get("title", work_metadata["id"]),
                    chunk_index=chunk_index,
                ))
                chunk_index += 1
        logger.info(f"Created {len(chunks)} chunks for {work_metadata.get('title', work_metadata['id'])}")
        return chunks

    def _split_text_into_chunks(self, text: str) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            if end < len(text):
                search_start = max(end - 100, start)
                sentence_end = max(
                    text.rfind(". ", search_start, end),
                    text.rfind("! ", search_start, end),
                    text.rfind("? ", search_start, end),
                )
                if sentence_end > start:
                    end = sentence_end + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - self.chunk_overlap
            if start >= end:
                start = end

        return chunks

    def _save_processed_document(self, doc: ProcessedDocument) -> None:
        output_file = self.output_path / f"{doc.work_id}.json"
        doc_data = {
            "work_id": doc.work_id,
            "title": doc.title,
            "author": doc.author,
            "year": doc.year,
            "genre": doc.genre,
            "total_pages": doc.total_pages,
            "total_text_length": doc.total_text_length,
            "chunks": [
                {
                    "text": chunk.text,
                    "source": chunk.source,
                    "page": chunk.page,
                    "work_id": chunk.work_id,
                    "author": chunk.author,
                    "title": chunk.title,
                    "chunk_index": chunk.chunk_index,
                }
                for chunk in doc.chunks
            ],
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(doc_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved processed document: {output_file}")

    def load_processed_document(self, work_id: str) -> Optional[ProcessedDocument]:
        file_path = self.output_path / f"{work_id}.json"
        if not file_path.exists():
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            chunks = [TextChunk(**chunk_data) for chunk_data in data["chunks"]]
            return ProcessedDocument(
                work_id=data["work_id"],
                title=data["title"],
                author=data["author"],
                year=data["year"],
                genre=data["genre"],
                chunks=chunks,
                total_pages=data["total_pages"],
                total_text_length=data["total_text_length"],
            )
        except Exception as e:
            logger.error(f"Error loading processed document {work_id}: {str(e)}")
            return None

    def get_all_chunks(self) -> List[TextChunk]:
        all_chunks: List[TextChunk] = []
        for doc in self._documents():
            processed = self.load_processed_document(doc.get("id", ""))
            if processed:
                all_chunks.extend(processed.chunks)
        return all_chunks


def main():
    processor = DocumentProcessor()
    logger.info("Starting document processing...")
    processed_docs = processor.process_all_documents()
    print(f"\nProcessed {len(processed_docs)} documents:")
    for doc in processed_docs:
        print(f"- {doc.title}: {len(doc.chunks)} chunks, {doc.total_pages} pages")


if __name__ == "__main__":
    main()
