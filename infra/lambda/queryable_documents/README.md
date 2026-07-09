# Queryable Documents Lambda

This Lambda powers the API Gateway `GET /documents/queryable` endpoint.

It reads metadata from `s3://<bucket>/embeddings/faiss/*.metadata.json` and returns the same response shape as the FastAPI `/documents/queryable` endpoint.

Gradio can use it by setting the environment variable:
```bash
QUERYABLE_API_URL="https://<api-id>.execute-api.<region>.amazonaws.com/documents/queryable"
```
