#!/usr/bin/env python3
"""
Book Data Splitter for RAG Systems
Splits PDF documents into smaller chunks for vector database ingestion.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Import required libraries
try:
    from pypdf import PdfReader
except ImportError:
    try:
        import PyPDF2 as pypdf
        PdfReader = pypdf.PdfReader
    except ImportError:
        raise ImportError("pypdf or PyPDF2 is required. Install with: pip install pypdf")

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.schema import Document
import json


@dataclass
class SplitterConfig:
    """Configuration for document splitting."""

    # Chunking parameters
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Splitting strategy
    split_strategy: str = "recursive"  # "recursive", "character", "sentence"

    # Input/Output
    input_dir: str = "books"
    output_dir: str = "chunks"
    output_format: str = "json"  # "json", "jsonl"

    # Processing options
    include_metadata: bool = True
    preserve_pages: bool = True

    # Performance
    batch_size: int = 10


class BookDataSplitter:
    """Main class for splitting book data into chunks."""

    def __init__(self, config: SplitterConfig):
        self.config = config
        self.text_splitter = self._create_text_splitter()

    def _create_text_splitter(self):
        """Create appropriate text splitter based on configuration."""
        if self.config.split_strategy == "recursive":
            return RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        elif self.config.split_strategy == "character":
            return CharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separator="\n\n"
            )
        else:
            raise ValueError(f"Unknown split strategy: {self.config.split_strategy}")

    def load_pdf(self, pdf_path: str) -> Tuple[str, Dict]:
        """Load a PDF file and extract text and metadata."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf = PdfReader(file)
                text = ""
                metadata = {}

                # Extract text from all pages with better error handling
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            # Clean the page text immediately to handle encoding issues
                            page_text = self._robust_clean_text(page_text)
                            if page_text.strip():  # Only add non-empty text
                                if self.config.preserve_pages:
                                    text += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"
                                else:
                                    text += f"\n\n{page_text}"
                    except Exception as page_error:
                        logging.warning(f"Failed to extract text from page {page_num + 1} in {pdf_path}: {str(page_error)}")
                        # Try alternative extraction methods
                        try:
                            # Try to get text using different extraction methods
                            if hasattr(page, 'get_contents'):
                                content = page.get_contents()
                                if content:
                                    # Extract text from content streams if available
                                    pass  # Skip for now, but could implement
                        except:
                            pass
                        continue

                # Final cleaning of the text
                text = self._robust_clean_text(text)

                # Extract metadata with error handling
                try:
                    if pdf.metadata:
                        metadata = {
                            'title': self._safe_str(pdf.metadata.get('/Title', '')),
                            'author': self._safe_str(pdf.metadata.get('/Author', '')),
                            'subject': self._safe_str(pdf.metadata.get('/Subject', '')),
                            'creator': self._safe_str(pdf.metadata.get('/Creator', '')),
                            'producer': self._safe_str(pdf.metadata.get('/Producer', '')),
                            'creation_date': self._safe_str(pdf.metadata.get('/CreationDate', '')),
                            'modification_date': self._safe_str(pdf.metadata.get('/ModDate', '')),
                            'total_pages': len(pdf.pages),
                            'file_path': pdf_path,
                            'file_name': Path(pdf_path).name
                        }
                except Exception as meta_error:
                    logging.warning(f"Failed to extract metadata from {pdf_path}: {str(meta_error)}")
                    metadata = {
                        'total_pages': len(pdf.pages),
                        'file_path': pdf_path,
                        'file_name': Path(pdf_path).name,
                        'title': Path(pdf_path).stem  # Use filename as fallback title
                    }

                return text, metadata

        except Exception as e:
            logging.error(f"Error loading PDF {pdf_path}: {str(e)}")
            # Return minimal data instead of raising exception
            return "", {
                'total_pages': 0,
                'file_path': pdf_path,
                'file_name': Path(pdf_path).name,
                'error': str(e)
            }

    def _safe_str(self, value) -> str:
        """Safely convert a value to string, handling encoding issues."""
        if value is None:
            return ''

        try:
            return str(value)
        except (UnicodeEncodeError, UnicodeDecodeError):
            try:
                # Try to encode and decode to clean the string
                return str(value).encode('utf-8', errors='replace').decode('utf-8')
            except:
                return f"<encoding_error: {type(value).__name__}>"

    def _clean_text(self, text: str) -> str:
        """Clean text to remove problematic Unicode characters."""
        if not text:
            return text

        # Remove surrogate characters that can't be encoded
        try:
            # First try to encode and decode to see if there are issues
            text.encode('utf-8')
            return text
        except UnicodeEncodeError:
            # If there are encoding issues, try to clean them
            cleaned_chars = []
            for char in text:
                try:
                    char.encode('utf-8')
                    cleaned_chars.append(char)
                except UnicodeEncodeError:
                    # Replace problematic characters with space
                    cleaned_chars.append(' ')

            return ''.join(cleaned_chars)

    def _robust_clean_text(self, text: str) -> str:
        """More robust text cleaning for problematic PDFs."""
        if not text:
            return text

        # Handle specific encoding errors
        try:
            # Try to normalize the text
            import unicodedata
            normalized = unicodedata.normalize('NFKD', text)
            return ''.join([c for c in normalized if not unicodedata.combining(c)])
        except:
            pass

        # Fallback to basic cleaning
        return self._clean_text(text)

    def process_book(self, pdf_path: str) -> List[Document]:
        """Process a single book PDF and return chunks as Document objects."""
        logging.info(f"Processing book: {pdf_path}")

        # Load PDF
        text, metadata = self.load_pdf(pdf_path)

        if not text.strip():
            logging.warning(f"No text extracted from {pdf_path}")
            return []

        # Create initial document
        doc_metadata = metadata.copy()
        if self.config.include_metadata:
            doc_metadata.update({
                'chunk_size': self.config.chunk_size,
                'chunk_overlap': self.config.chunk_overlap,
                'split_strategy': self.config.split_strategy
            })

        # Create Document object
        document = Document(
            page_content=text,
            metadata=doc_metadata
        )

        # Split the document
        chunks = self.text_splitter.split_documents([document])

        # Add chunk-specific metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                'chunk_index': i,
                'total_chunks': len(chunks)
            })

        logging.info(f"Created {len(chunks)} chunks from {pdf_path}")
        return chunks

    def process_directory(self) -> Dict[str, List[Document]]:
        """Process all PDF files in the input directory."""
        input_path = Path(self.config.input_dir)

        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_path}")

        pdf_files = list(input_path.glob("*.pdf"))
        logging.info(f"Found {len(pdf_files)} PDF files to process")

        if not pdf_files:
            logging.warning(f"No PDF files found in {input_path}")
            return {}

        all_chunks = {}
        processed_count = 0

        for pdf_file in pdf_files:
            try:
                chunks = self.process_book(str(pdf_file))
                all_chunks[pdf_file.name] = chunks
                processed_count += 1

                # Progress logging
                if processed_count % self.config.batch_size == 0:
                    logging.info(f"Processed {processed_count}/{len(pdf_files)} books")

            except Exception as e:
                logging.error(f"Failed to process {pdf_file.name}: {str(e)}")
                continue

        logging.info(f"Successfully processed {processed_count}/{len(pdf_files)} books")
        return all_chunks


def setup_logging():
    """Setup logging configuration."""
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Setup file handler with UTF-8 encoding
    file_handler = logging.FileHandler('book_splitter.log', encoding='utf-8')
    file_handler.setFormatter(file_formatter)

    # Setup console handler with proper encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )


def parse_arguments() -> SplitterConfig:
    """Parse command line arguments and return configuration."""
    parser = argparse.ArgumentParser(description="Split book PDFs into chunks for RAG")

    parser.add_argument("--input-dir", default="books",
                       help="Directory containing PDF files")
    parser.add_argument("--output-dir", default="chunks",
                       help="Directory to save chunk files")
    parser.add_argument("--chunk-size", type=int, default=1000,
                       help="Size of each chunk in characters")
    parser.add_argument("--chunk-overlap", type=int, default=200,
                       help="Overlap between chunks in characters")
    parser.add_argument("--split-strategy", choices=["recursive", "character"],
                       default="recursive", help="Text splitting strategy")
    parser.add_argument("--output-format", choices=["json", "jsonl"],
                       default="json", help="Output format")
    parser.add_argument("--no-metadata", action="store_true",
                       help="Exclude metadata from output")
    parser.add_argument("--no-pages", action="store_true",
                       help="Don't preserve page information")

    args = parser.parse_args()

    return SplitterConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        split_strategy=args.split_strategy,
        output_format=args.output_format,
        include_metadata=not args.no_metadata,
        preserve_pages=not args.no_pages
    )


def save_chunks(chunks_dict: Dict[str, List[Document]], output_dir: str, output_format: str):
    """Save chunks to files in the specified format."""
    output_path = Path(output_dir)

    if output_format == "json":
        # Save as a single JSON file with all chunks organized by book
        output_file = output_path / "all_chunks.json"

        json_data = {}
        total_chunks = 0

        for book_name, chunks in chunks_dict.items():
            book_chunks = []
            for chunk in chunks:
                # Ensure content is properly encoded
                content = chunk.page_content
                if isinstance(content, str):
                    # Use robust cleaning to remove problematic characters
                    content = _clean_unicode_for_json(content)

                chunk_dict = {
                    'content': content,
                    'metadata': chunk.metadata
                }
                book_chunks.append(chunk_dict)
            json_data[book_name] = book_chunks
            total_chunks += len(chunks)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=True, indent=2, default=str)

        logging.info(f"Saved {total_chunks} total chunks to {output_file}")

    elif output_format == "jsonl":
        # Save as JSONL format (one chunk per line)
        output_file = output_path / "all_chunks.jsonl"

        with open(output_file, 'w', encoding='utf-8') as f:
            total_chunks = 0

            for book_name, chunks in chunks_dict.items():
                for chunk in chunks:
                    # Clean content for JSON serialization
                    content = chunk.page_content
                    if isinstance(content, str):
                        content = _clean_unicode_for_json(content)

                    chunk_dict = {
                        'book_name': book_name,
                        'content': content,
                        'metadata': chunk.metadata
                    }

                    # Use ensure_ascii=True to avoid encoding issues when reading back
                    json_str = json.dumps(chunk_dict, ensure_ascii=True, default=str)
                    f.write(json_str + '\n')
                    total_chunks += 1

        logging.info(f"Saved {total_chunks} total chunks to {output_file}")

    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def _clean_unicode_for_json(text: str) -> str:
    """Clean Unicode text for JSON serialization."""
    if not text:
        return text

    # Use the robust cleaning method
    try:
        # Test if the text can be JSON serialized with ensure_ascii=True
        json.dumps(text, ensure_ascii=True)
        return text
    except (UnicodeEncodeError, UnicodeDecodeError, TypeError):
        # If there are issues, use robust cleaning
        cleaned = []
        for char in text:
            try:
                json.dumps(char, ensure_ascii=True)
                cleaned.append(char)
            except (UnicodeEncodeError, UnicodeDecodeError, TypeError):
                # Replace problematic characters with space
                cleaned.append(' ')

        return ''.join(cleaned)


def main():
    """Main execution function."""
    setup_logging()
    config = parse_arguments()

    logging.info("Starting Book Data Splitter with configuration:")
    logging.info(f"Input directory: {config.input_dir}")
    logging.info(f"Output directory: {config.output_dir}")
    logging.info(f"Chunk size: {config.chunk_size}")
    logging.info(f"Chunk overlap: {config.chunk_overlap}")
    logging.info(f"Split strategy: {config.split_strategy}")
    logging.info(f"Output format: {config.output_format}")

    splitter = BookDataSplitter(config)

    # Create output directory if it doesn't exist
    Path(config.output_dir).mkdir(exist_ok=True)

    try:
        # Process all books in the input directory
        logging.info("Starting PDF processing...")
        chunks_dict = splitter.process_directory()

        if not chunks_dict:
            logging.warning("No chunks were generated. Check input directory and PDF files.")
            return

        # Save chunks to output files
        logging.info("Saving chunks to output files...")
        save_chunks(chunks_dict, config.output_dir, config.output_format)

        # Print summary
        total_books = len(chunks_dict)
        total_chunks = sum(len(chunks) for chunks in chunks_dict.values())

        logging.info("Processing completed successfully!")
        logging.info(f"Processed {total_books} books")
        logging.info(f"Generated {total_chunks} total chunks")

        # Show file locations
        output_path = Path(config.output_dir)
        if config.output_format == "json":
            output_file = output_path / "all_chunks.json"
            logging.info(f"Output saved to: {output_file}")
        else:
            output_file = output_path / "all_chunks.jsonl"
            logging.info(f"Output saved to: {output_file}")

    except Exception as e:
        logging.error(f"Error during processing: {str(e)}")
        raise


if __name__ == "__main__":
    main()