import argparse
import json
import logging
import os
import datetime
from pathlib import Path
from typing import Optional

import boto3
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants and Macro Variables ---
# CHUNK_SIZE controls the maximum length of a string before it's cut (impacts LLM context window limits).
CHUNK_SIZE = 1000

# CHUNK_OVERLAP ensures that the end of one chunk is repeated at the beginning of the next chunk 
# so context isn't lost if a sentence is split exactly at the boundary.
CHUNK_OVERLAP = 200

# The standard model for embedding generation (must match the backend API's expected model).
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# The output dimension size for the chosen embedding model (all-MiniLM-L6-v2 outputs a 384-dimensional vector).
EMBEDDING_DIMENSION = 384


def safe_name_from_key(key: str) -> str:
    """Generates a safe base filename from an S3 key."""
    base_name = key.split('/')[-1]
    # Remove .pdf extension if present
    if base_name.lower().endswith('.pdf'):
        base_name = base_name[:-4]
    # Keep alphanumeric, replace others with underscore
    safe_name = "".join([c if c.isalnum() else "_" for c in base_name])
    return safe_name


def download_pdf_from_s3(s3_client, bucket: str, key: str, version_id: Optional[str], download_path: str):
    """Downloads a PDF from S3."""
    logger.info(f"Downloading PDF from s3://{bucket}/{key} (version: {version_id})")
    extra_args = {}
    if version_id:
        extra_args['VersionId'] = version_id
        
    try:
        s3_client.download_file(bucket, key, download_path, ExtraArgs=extra_args)
        logger.info(f"Successfully downloaded to {download_path}")
    except Exception as e:
        logger.error(f"Failed to download from S3: {e}")
        raise


def extract_pages_text(pdf_path: str) -> list:
    """Extracts text page by page using pypdf."""
    logger.info(f"Extracting text from {pdf_path}")
    reader = PdfReader(pdf_path)
    pages = []
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            # Clean up text by removing excessive whitespaces
            text = " ".join(text.split())
            if text:
                pages.append({"page_num": i + 1, "text": text})
            
    if not pages:
        raise ValueError("PDF has no extractable text or is completely empty.")
        
    logger.info(f"Extracted {len(pages)} pages with text.")
    return pages


def chunk_pages(pages: list, bucket: str, key: str, version_id: Optional[str], safe_name: str) -> list:
    """
    Complex Logic: Text Chunking (Sliding Window Approach)
    Breaks large continuous page texts into smaller, overlapping chunks.
    The overlapping window prevents sentences/context from being hard-cut at the CHUNK_SIZE boundary.
    """
    logger.info("Chunking text pages...")
    chunks = []
    chunk_index = 1
    
    for page in pages:
        text = page["text"]
        page_num = page["page_num"]
        
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + CHUNK_SIZE
            chunk_text = text[start:end]
            
            chunk_id = f"{safe_name}_chunk_{chunk_index:04d}"
            
            chunk_record = {
                "chunk_id": chunk_id,
                "source_bucket": bucket,
                "source_key": key,
                "source_version_id": version_id,
                "page_start": page_num,
                "page_end": page_num,
                "text": chunk_text,
                "text_preview": chunk_text[:250]
            }
            chunks.append(chunk_record)
            
            chunk_index += 1
            start += (CHUNK_SIZE - CHUNK_OVERLAP)
            
    logger.info(f"Generated {len(chunks)} chunks.")
    return chunks


def generate_embeddings(chunks: list) -> np.ndarray:
    """Generates embeddings using sentence-transformers."""
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    logger.info(f"Generating embeddings for {len(chunks)} chunks...")
    texts = [chunk["text"] for chunk in chunks]
    
    # We use show_progress_bar=False to avoid excessive stdout on EC2 background tasks,
    # but the prompt asked for "progress logs" - the logger.info is adequate.
    embeddings = model.encode(texts, show_progress_bar=False)
    embeddings = np.array(embeddings).astype('float32')
    
    logger.info(f"Generated embeddings shape: {embeddings.shape}")
    return embeddings


def build_faiss_index(embeddings: np.ndarray, index_path: str):
    """Builds and saves a FAISS IndexFlatL2 index."""
    logger.info("Building FAISS IndexFlatL2 index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    logger.info(f"Saving FAISS index to {index_path}")
    faiss.write_index(index, index_path)


def upload_file_to_s3(s3_client, local_path: str, bucket: str, s3_key: str):
    """Uploads a local file to S3."""
    logger.info(f"Uploading file to s3://{bucket}/{s3_key}")
    s3_client.upload_file(local_path, bucket, s3_key)


def upload_json_to_s3(s3_client, data: dict, bucket: str, s3_key: str):
    """Uploads a python dictionary to S3 as a JSON object."""
    logger.info(f"Uploading JSON data to s3://{bucket}/{s3_key}")
    s3_client.put_object(
        Bucket=bucket,
        Key=s3_key,
        Body=json.dumps(data, indent=2),
        ContentType='application/json'
    )


def process_pdf_to_faiss(
    bucket: str,
    key: str,
    version_id: Optional[str] = None,
    output_name: Optional[str] = None
) -> dict:
    """
    Complex Pipeline Orchestration:
    This function runs the entire end-to-end embedding pipeline for a single document:
    1. Download PDF: Fetches raw PDF from S3 to local storage.
    2. Extract Text: Parses PDF to raw strings.
    3. Chunk Text: Slices string into overlapping tokens/blocks.
    4. Generate Embeddings: Passes chunks through the SentenceTransformer model to create vectors.
    5. Build FAISS Index: Inserts vectors into an L2 Distance FAISS index.
    6. Upload to S3: Stores the chunks (.jsonl), the binary FAISS index (.index), and metadata back to S3.
    """
    safe_name = output_name if output_name else safe_name_from_key(key)
    logger.info(f"Using base name for outputs: {safe_name}")
    
    work_dir = Path("./work")
    work_dir.mkdir(parents=True, exist_ok=True)
    
    local_pdf_path = str(work_dir / f"{safe_name}.pdf")
    local_chunks_path = str(work_dir / f"{safe_name}.chunks.jsonl")
    local_index_path = str(work_dir / f"{safe_name}.index")
    
    s3_client = boto3.client('s3')
    
    try:
        # 1. Download PDF
        download_pdf_from_s3(s3_client, bucket, key, version_id, local_pdf_path)
        
        # 2. Extract Text
        pages = extract_pages_text(local_pdf_path)
        
        # 3. Chunk Text
        chunks = chunk_pages(pages, bucket, key, version_id, safe_name)
        
        # Save chunks to local JSONL
        with open(local_chunks_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk) + '\n')
                
        # 4. Generate Embeddings
        embeddings = generate_embeddings(chunks)
        
        # 5. Build FAISS Index
        build_faiss_index(embeddings, local_index_path)
        
        # 6. Upload Outputs to S3
        chunks_s3_key = f"processed-chunks/{safe_name}.chunks.jsonl"
        faiss_index_s3_key = f"embeddings/faiss/{safe_name}.index"
        faiss_metadata_s3_key = f"embeddings/faiss/{safe_name}.metadata.json"
        embedding_status_s3_key = f"processed-metadata/{safe_name}.embedding-status.json"
        
        now_str = datetime.datetime.utcnow().isoformat() + "Z"
        
        faiss_metadata = {
            "source_bucket": bucket,
            "source_key": key,
            "source_version_id": version_id,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "chunk_count": len(chunks),
            "faiss_index_type": "IndexFlatL2",
            "chunks_s3_key": chunks_s3_key,
            "faiss_index_s3_key": faiss_index_s3_key,
            "created_at": now_str,
            "status": "embedded"
        }
        
        embedding_status = {
            "source_bucket": bucket,
            "source_key": key,
            "source_version_id": version_id,
            "pipeline_stage": "embedding_completed",
            "rag_status": "embedded",
            "chunks_s3_key": chunks_s3_key,
            "faiss_index_s3_key": faiss_index_s3_key,
            "embedding_metadata_s3_key": faiss_metadata_s3_key,
            "chunk_count": len(chunks),
            "created_at": now_str
        }
        
        upload_file_to_s3(s3_client, local_chunks_path, bucket, chunks_s3_key)
        upload_file_to_s3(s3_client, local_index_path, bucket, faiss_index_s3_key)
        
        upload_json_to_s3(s3_client, faiss_metadata, bucket, faiss_metadata_s3_key)
        upload_json_to_s3(s3_client, embedding_status, bucket, embedding_status_s3_key)
        
        logger.info("Pipeline completed successfully.")
        
        return {
            "safe_name": safe_name,
            "chunks_s3_key": chunks_s3_key,
            "faiss_index_s3_key": faiss_index_s3_key,
            "faiss_metadata_s3_key": faiss_metadata_s3_key,
            "embedding_status_s3_key": embedding_status_s3_key,
            "chunk_count": len(chunks),
            "status": "embedded"
        }
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Process PDF to FAISS chunks and embeddings for RAG.")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--key", required=True, help="S3 object key (e.g., raw-pdfs/doc.pdf)")
    parser.add_argument("--version-id", default=None, help="S3 object version ID (optional)")
    parser.add_argument("--output-name", default=None, help="Custom output name (optional)")
    
    args = parser.parse_args()
    
    try:
        process_pdf_to_faiss(
            bucket=args.bucket,
            key=args.key,
            version_id=args.version_id,
            output_name=args.output_name
        )
    except Exception as e:
        logger.error(f"CLI execution failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
