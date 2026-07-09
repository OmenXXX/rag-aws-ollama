import os
import json
import logging
from typing import Optional, Dict, Any, List

import boto3
import faiss
import numpy as np
import requests
from botocore.exceptions import ClientError
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from process_pdf_to_faiss import process_pdf_to_faiss, safe_name_from_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="AWS RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BUCKET_NAME = os.environ.get("RAG_BUCKET", "ayan-deaws-lab-600743178533")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "tinyllama")

s3_client = boto3.client("s3")
_embedding_model: Optional[SentenceTransformer] = None


class QueryRequest(BaseModel):
    doc_name: str
    question: str
    top_k: int = 5


class RagAnswerRequest(BaseModel):
    doc_name: str
    question: str
    top_k: int = 5
    model: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok", "service": "aws-rag-processor"}


@app.post("/documents/upload")
def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    auto_ingest: bool = Form(True),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    safe_name = safe_name_from_key(file.filename)
    s3_key = f"raw-pdfs/{safe_name}.pdf"

    try:
        work_dir = "./work/api/"
        os.makedirs(work_dir, exist_ok=True)
        local_path = os.path.join(work_dir, os.path.basename(file.filename))

        with open(local_path, "wb") as buffer:
            buffer.write(file.file.read())

        with open(local_path, "rb") as data:
            response = s3_client.put_object(
                Bucket=BUCKET_NAME,
                Key=s3_key,
                Body=data,
                ContentType="application/pdf",
            )

        version_id = response.get("VersionId")

        if auto_ingest:
            background_tasks.add_task(
                process_pdf_to_faiss,
                BUCKET_NAME,
                s3_key,
                version_id,
                safe_name,
            )

        return {
            "message": "uploaded",
            "bucket": BUCKET_NAME,
            "key": s3_key,
            "version_id": version_id,
            "safe_name": safe_name,
            "auto_ingest_started": auto_ingest,
        }
    except Exception as e:
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents/queryable")
def list_queryable_documents():
    try:
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix="embeddings/faiss/")
        if "Contents" not in response:
            return []

        metadata_files = [obj for obj in response["Contents"] if obj["Key"].endswith(".metadata.json")]
        metadata_files.sort(key=lambda x: x["LastModified"], reverse=True)

        results = []
        for obj in metadata_files:
            try:
                res = s3_client.get_object(Bucket=BUCKET_NAME, Key=obj["Key"])
                data = json.loads(res["Body"].read().decode("utf-8"))
                doc_name = obj["Key"].split("/")[-1].replace(".metadata.json", "")

                results.append(
                    {
                        "doc_name": doc_name,
                        "source_key": data.get("source_key"),
                        "source_version_id": data.get("source_version_id"),
                        "chunk_count": data.get("chunk_count"),
                        "embedding_model": data.get("embedding_model"),
                        "chunks_s3_key": data.get("chunks_s3_key"),
                        "faiss_index_s3_key": data.get("faiss_index_s3_key"),
                        "created_at": data.get("created_at"),
                        "status": data.get("status"),
                    }
                )
            except Exception as e:
                logger.warning("Failed to read metadata %s: %s", obj["Key"], e)

        return results
    except Exception as e:
        logger.exception("List failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents/status/{doc_name}")
def get_document_status(doc_name: str):
    s3_key = f"processed-metadata/{doc_name}.embedding-status.json"
    try:
        res = s3_client.get_object(Bucket=BUCKET_NAME, Key=s3_key)
        return json.loads(res["Body"].read().decode("utf-8"))
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ["NoSuchKey", "404"]:
            return {"doc_name": doc_name, "status": "not_embedded"}
        logger.exception("Status check failed")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Status check failed")
        raise HTTPException(status_code=500, detail=str(e))


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading query embedding model: %s", EMBEDDING_MODEL)
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def retrieve_chunks_for_question(doc_name: str, question: str, top_k: int) -> Dict[str, Any]:
    """Load a FAISS index and chunk file from S3, then return the top matching chunks."""
    if top_k < 1:
        top_k = 1
    if top_k > 20:
        top_k = 20

    work_dir = "./work/api/"
    os.makedirs(work_dir, exist_ok=True)

    index_s3_key = f"embeddings/faiss/{doc_name}.index"
    chunks_s3_key = f"processed-chunks/{doc_name}.chunks.jsonl"

    local_index_path = os.path.join(work_dir, f"{doc_name}.index")
    local_chunks_path = os.path.join(work_dir, f"{doc_name}.chunks.jsonl")

    try:
        if not os.path.exists(local_index_path):
            logger.info("Downloading FAISS index: s3://%s/%s", BUCKET_NAME, index_s3_key)
            s3_client.download_file(BUCKET_NAME, index_s3_key, local_index_path)

        if not os.path.exists(local_chunks_path):
            logger.info("Downloading chunks: s3://%s/%s", BUCKET_NAME, chunks_s3_key)
            s3_client.download_file(BUCKET_NAME, chunks_s3_key, local_chunks_path)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ["NoSuchKey", "404"]:
            raise HTTPException(status_code=404, detail="Document not found or not fully processed.")
        raise

    index = faiss.read_index(local_index_path)

    chunks: List[Dict[str, Any]] = []
    with open(local_chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    if not chunks:
        raise HTTPException(status_code=404, detail="No chunks found for this document.")

    model = _get_embedding_model()
    q_emb = model.encode([question], show_progress_bar=False)
    q_emb = np.array(q_emb).astype("float32")

    distances, indices = index.search(q_emb, min(top_k, len(chunks)))

    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1 and idx < len(chunks):
            chunk = chunks[idx]
            results.append(
                {
                    "rank": i + 1,
                    "score": float(distances[0][i]),
                    "chunk_id": chunk.get("chunk_id"),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "text_preview": chunk.get("text_preview"),
                    "text": chunk.get("text"),
                }
            )

    return {"doc_name": doc_name, "question": question, "top_k": top_k, "results": results}


@app.post("/rag/query")
def query_rag(req: QueryRequest):
    try:
        data = retrieve_chunks_for_question(req.doc_name, req.question, req.top_k)
        data["answer_note"] = "This endpoint returns retrieved context chunks only. Use /rag/answer for Ollama answer generation."
        return data
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))


def build_rag_user_message(question: str, chunks: List[Dict[str, Any]]) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks[:3]):
        text = (chunk.get("text") or "").strip()
        if len(text) > 700:
            text = text[:700] + "..."
        context_blocks.append(
            f"[Evidence {i + 1} | Page {chunk.get('page_start')}]\n{text}"
        )

    context = "\n\n".join(context_blocks)
    
    q_lower = question.lower()
    extra_instructions = ""
    if "candidate name" in q_lower or "name of candidate" in q_lower or "who is the candidate" in q_lower:
        extra_instructions = "\nReturn only the person's name."
    elif "technical skills" in q_lower or "skills" in q_lower:
        extra_instructions = "\nReturn a concise bullet list grouped by category."

    return f"""Question:
{question}{extra_instructions}

Evidence:
{context}

Return the answer only."""


@app.post("/rag/answer")
def answer_rag(req: RagAnswerRequest):
    selected_model = req.model or OLLAMA_MODEL
    try:
        retrieval = retrieve_chunks_for_question(req.doc_name, req.question, req.top_k)
        chunks = retrieval.get("results", [])
        user_msg = build_rag_user_message(req.question, chunks)

        try:
            logger.info("Calling Ollama model %s at %s", selected_model, OLLAMA_URL)
            response = requests.post(
                f"{OLLAMA_URL.rstrip('/')}/api/chat",
                json={
                    "model": selected_model, 
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a strict document QA assistant. Answer only from the provided evidence. Do not repeat the prompt. Do not repeat the evidence. Do not mention context chunks. Give only the final answer. If the answer is not present, say: The document does not contain enough information."
                        },
                        {
                            "role": "user",
                            "content": user_msg
                        }
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                        "num_predict": 250,
                        "num_ctx": 2048
                    }
                },
                timeout=120,
            )
            response.raise_for_status()
            ollama_data = response.json()
            answer = (ollama_data.get("message", {}).get("content") or "").strip()
            
            if answer:
                if "Final answer:" in answer:
                    answer = answer.split("Final answer:")[-1].strip()
                if "Evidence:" in answer:
                    answer = answer.split("Evidence:")[-1].strip()
                if "Question:" in answer:
                    answer = answer.split("Question:")[-1].strip()
            
            if not answer:
                answer = "Ollama returned an empty response. Retrieved chunks are returned below."
        except Exception as ollama_error:
            logger.exception("Ollama answer generation failed")
            answer = f"Ollama answer generation failed: {ollama_error}. Retrieved chunks are returned below."

        return {
            "doc_name": req.doc_name,
            "question": req.question,
            "model": selected_model,
            "answer": answer,
            "retrieved_chunks": chunks,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Answer generation failed")
        raise HTTPException(status_code=500, detail=str(e))
