import gradio as gr
import requests
import os

API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")
QUERYABLE_API_URL = os.environ.get("QUERYABLE_API_URL")


def upload_pdf(file_path, auto_ingest):
    if not file_path:
        return "No file selected."
    try:
        with open(file_path, "rb") as f:
            files = {"file": f}
            data = {"auto_ingest": auto_ingest}
            response = requests.post(f"{API_BASE}/documents/upload", files=files, data=data, timeout=30)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        return f"Error: {str(e)}"


def get_queryable_documents():
    try:
        url = QUERYABLE_API_URL or f"{API_BASE}/documents/queryable"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        docs = response.json()
        doc_names = [doc["doc_name"] for doc in docs]
        return gr.update(choices=doc_names, value=doc_names[0] if doc_names else None), docs
    except Exception as e:
        return gr.update(choices=[], value=None), f"Error: {str(e)}"


def _format_chunks(chunks):
    output = ""
    for res in chunks:
        score = res.get("score")
        score_text = f"{score:.4f}" if isinstance(score, (float, int)) else str(score)
        output += f"Rank: {res.get('rank')} | Score: {score_text} | Page: {res.get('page_start')}\n"
        output += f"Preview: {res.get('text_preview')}...\n"
        output += f"Full Text: {res.get('text')}\n"
        output += "-" * 50 + "\n"
    return output


def query_rag(doc_name, question, top_k, generate_answer, ollama_model):
    if not doc_name:
        return "", "Please select a document."
    if not question:
        return "", "Please enter a question."

    try:
        payload = {"doc_name": doc_name, "question": question, "top_k": int(top_k)}

        if generate_answer:
            payload["model"] = ollama_model or "qwen2.5:1.5b"
            response = requests.post(f"{API_BASE}/rag/answer", json=payload, timeout=180)
            response.raise_for_status()
            data = response.json()
            answer = data.get("answer", "")
            chunks_text = _format_chunks(data.get("retrieved_chunks", []))
            return answer, chunks_text

        response = requests.post(f"{API_BASE}/rag/query", json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        answer = data.get("answer_note", "Retrieved chunks only.")
        chunks_text = _format_chunks(data.get("results", []))
        return answer, chunks_text
    except Exception as e:
        return "", f"Error: {str(e)}"


with gr.Blocks(title="AWS RAG Processor UI") as demo:
    gr.Markdown("# Minimal EC2-hosted RAG")

    with gr.Tab("Upload PDF"):
        file_input = gr.File(label="Upload PDF")
        auto_ingest_cb = gr.Checkbox(label="Auto ingest after upload", value=True)
        upload_btn = gr.Button("Upload")
        upload_output = gr.JSON(label="Upload Response")

        upload_btn.click(upload_pdf, inputs=[file_input, auto_ingest_cb], outputs=[upload_output])

    with gr.Tab("Query Documents"):
        with gr.Row():
            refresh_btn = gr.Button("Refresh queryable documents")
            doc_dropdown = gr.Dropdown(label="Select Document", choices=[])

        doc_metadata_output = gr.JSON(label="Available Documents Metadata")

        refresh_btn.click(get_queryable_documents, inputs=[], outputs=[doc_dropdown, doc_metadata_output])

        question_input = gr.Textbox(label="Question")
        top_k_slider = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Top K")
        generate_answer_cb = gr.Checkbox(label="Generate final answer with Ollama", value=True)
        ollama_model_input = gr.Textbox(label="Ollama model", value="qwen2.5:1.5b")
        query_btn = gr.Button("Query Selected PDF")

        answer_output = gr.Textbox(label="Final Answer", lines=8)
        chunks_output = gr.Textbox(label="Retrieved Evidence Chunks", lines=15)

        query_btn.click(
            query_rag,
            inputs=[doc_dropdown, question_input, top_k_slider, generate_answer_cb, ollama_model_input],
            outputs=[answer_output, chunks_output],
        )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
