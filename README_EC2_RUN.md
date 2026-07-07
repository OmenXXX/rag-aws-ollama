# AWS RAG Processor for EC2

This processor downloads a PDF from S3, extracts text, chunks it, generates embeddings using `sentence-transformers/all-MiniLM-L6-v2`, builds a FAISS index, and uploads the results back to S3. It also provides a FastAPI backend and a Gradio UI for interacting with the document.

## Setup Instructions (Ubuntu EC2)

1. Update system and install python3-venv and pip:
   ```bash
   sudo apt update
   sudo apt install -y python3-venv python3-pip
   ```

2. Clone or copy this project directory to your EC2 instance.

3. Navigate to the project directory:
   ```bash
   cd aws_rag_processor
   ```

4. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

5. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the API and UI

Ensure your EC2 instance has an IAM Role attached with S3 read/write permissions for the target bucket (do NOT use access keys in a `.env` file).

Run FastAPI (Backend):
```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```
*Note: Do not expose 8000 publicly for this POC.*

Run Gradio (Frontend) in another terminal:
```bash
python gradio_app.py
```

## Security Group Configuration

Open port `7860` only to **My IP** for Gradio access.

## Testing

1. Open Gradio at `http://<EC2_PUBLIC_IP>:7860`
2. Upload a PDF using the UI.
3. Wait for the background ingestion to complete.
4. Go to the "Query Documents" tab and click "Refresh queryable documents".
5. Select the PDF from the dropdown.
6. Ask a question!

*Note: The endpoint currently returns semantic search chunks. Full LLM answer generation can be added later.*

## Running the Processor Manually (CLI)

Example command:
```bash
python process_pdf_to_faiss.py \
  --bucket ayan-deaws-lab-600743178533 \
  --key raw-pdfs/GSLV_F16NISAR_Launch_Brochure.pdf \
  --version-id A1Yah1jIIt6aB7yFzUClsPMR.odMzRhw
```

## Important Reminder
**Remember to stop or terminate your EC2 instance after testing to avoid unnecessary AWS charges!**
