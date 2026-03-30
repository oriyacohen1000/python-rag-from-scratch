import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from main import (
    process_pdf_files,
    process_docx_files,
    get_chunks,
    create_vector_store,
    retrieve_k_relevant_chunks,
    generate_answer,
    read_questions_from_file,
)

st.set_page_config(page_title="RAG & Q&A Tool", page_icon="🔍", layout="wide")
st.title("🔍 RAG & Q&A Tool")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

mode = st.sidebar.radio("Mode", ["💬 RAG Chat", "📋 Batch Q&A"])


# ─── helpers ──────────────────────────────────────────────────────────────────

def save_uploaded_file(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def collect_file_paths(uploaded_files, path_input: str, supported_exts: list[str]) -> list[str]:
    """Merge drag-and-drop uploads and manual path inputs into one list of temp paths."""
    paths = []

    for uf in uploaded_files:
        ext = os.path.splitext(uf.name)[1].lower()
        if ext in supported_exts:
            paths.append(save_uploaded_file(uf))
        else:
            st.warning(f"Skipped '{uf.name}': unsupported format.")

    for raw in path_input.splitlines():
        raw = raw.strip().strip('"').strip("'")
        if not raw:
            continue
        if not os.path.isfile(raw):
            st.error(f"File not found: {raw}")
            continue
        ext = os.path.splitext(raw)[1].lower()
        if ext not in supported_exts:
            st.error(f"Unsupported format '{ext}': {raw}")
            continue
        paths.append(raw)

    return paths


def build_knowledge_base(all_paths: list[str], chunk_size: int, overlap: int):
    pdf_paths = [p for p in all_paths if p.lower().endswith(".pdf")]
    docx_paths = [p for p in all_paths if p.lower().endswith(".docx")]

    pages_data = []
    if pdf_paths:
        with st.spinner(f"Extracting text from {len(pdf_paths)} PDF(s)…"):
            pages_data += process_pdf_files(pdf_paths)
    if docx_paths:
        with st.spinner(f"Extracting text from {len(docx_paths)} DOCX file(s)…"):
            pages_data += process_docx_files(docx_paths)

    with st.spinner("Chunking and embedding…"):
        chunks = get_chunks(pages_data, chunk_size, overlap)
        database = create_vector_store(chunks)

    return database, len(chunks), len(pages_data)


# ─── Mode 1: RAG Chat ─────────────────────────────────────────────────────────

if mode == "💬 RAG Chat":
    st.header("RAG Chat")

    # ── Initial setup (shown only before knowledge base is built) ──
    if "database" not in st.session_state:
        st.subheader("Step 1 — Configure")
        col1, col2, col3 = st.columns(3)
        init_chunk = col1.number_input("Chunk Size", 100, 5000, 500, 50, key="init_chunk")
        init_overlap = col2.number_input("Overlap", 0, 500, 50, 10, key="init_overlap")
        init_k = col3.number_input("K (chunks to retrieve)", 1, 20, 3, 1, key="init_k")

        st.subheader("Step 2 — Upload Documents")
        uploaded = st.file_uploader(
            "Drag & drop your files here (.pdf / .docx)",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key="rag_upload",
        )
        path_input = st.text_area(
            "Or paste file paths (one per line)",
            placeholder="/path/to/document.pdf\n/path/to/report.docx",
            key="rag_paths",
        )

        if st.button("Build Knowledge Base", type="primary"):
            all_paths = collect_file_paths(uploaded or [], path_input or "", [".pdf", ".docx"])
            if not all_paths:
                st.error("Please provide at least one document.")
            else:
                database, n_chunks, n_pages = build_knowledge_base(all_paths, init_chunk, init_overlap)
                st.session_state["database"] = database
                st.session_state["all_paths"] = all_paths
                st.session_state["chunk_size"] = int(init_chunk)
                st.session_state["overlap"] = int(init_overlap)
                st.session_state["k_value"] = int(init_k)
                st.session_state["chat_history"] = []
                st.session_state["n_chunks"] = n_chunks
                st.session_state["n_pages"] = n_pages
                st.rerun()

    # ── Active chat UI (shown after knowledge base is built) ──
    else:
        # ── ⚙️ Settings panel (always visible) ──────────────────────────────
        with st.expander("⚙️ Settings", expanded=False):
            col1, col2, col3 = st.columns(3)
            new_k = col1.number_input(
                "K (chunks to retrieve)",
                1, 20,
                st.session_state["k_value"],
                1,
                key="live_k",
                help="Changes apply to the next question immediately — no rebuild needed.",
            )
            new_chunk = col2.number_input(
                "Chunk Size",
                100, 5000,
                st.session_state["chunk_size"],
                50,
                key="live_chunk",
                help="Changing this requires rebuilding the knowledge base.",
            )
            new_overlap = col3.number_input(
                "Overlap",
                0, 500,
                st.session_state["overlap"],
                10,
                key="live_overlap",
                help="Changing this requires rebuilding the knowledge base.",
            )

            # K updates instantly
            if int(new_k) != st.session_state["k_value"]:
                st.session_state["k_value"] = int(new_k)
                st.info(f"K updated to {new_k} — takes effect on your next question.")

            # Chunk / overlap changes require a rebuild
            chunk_changed = int(new_chunk) != st.session_state["chunk_size"]
            overlap_changed = int(new_overlap) != st.session_state["overlap"]
            if chunk_changed or overlap_changed:
                st.warning(
                    "Chunk Size or Overlap changed. Click **Rebuild** to re-process your documents with the new values."
                )
                if st.button("🔄 Rebuild Knowledge Base"):
                    database, n_chunks, n_pages = build_knowledge_base(
                        st.session_state["all_paths"], int(new_chunk), int(new_overlap)
                    )
                    st.session_state["database"] = database
                    st.session_state["chunk_size"] = int(new_chunk)
                    st.session_state["overlap"] = int(new_overlap)
                    st.session_state["n_chunks"] = n_chunks
                    st.session_state["n_pages"] = n_pages
                    st.success(f"Rebuilt! {n_chunks} chunks from {n_pages} page(s).")
                    st.rerun()

            st.caption(
                f"Current: Chunk={st.session_state['chunk_size']}  "
                f"Overlap={st.session_state['overlap']}  "
                f"K={st.session_state['k_value']}  "
                f"| {st.session_state['n_chunks']} chunks · {st.session_state['n_pages']} pages"
            )

        # ── Add Files ────────────────────────────────────────────────────────
        with st.expander("➕ Add Files to Knowledge Base", expanded=False):
            new_uploaded = st.file_uploader(
                "Drag & drop additional files (.pdf / .docx)",
                type=["pdf", "docx"],
                accept_multiple_files=True,
                key="add_upload",
            )
            new_path_input = st.text_area(
                "Or paste file paths (one per line)",
                placeholder="/path/to/more.pdf",
                key="add_paths",
            )
            if st.button("Add to Knowledge Base", type="primary", key="add_files_btn"):
                new_paths = collect_file_paths(new_uploaded or [], new_path_input or "", [".pdf", ".docx"])
                if not new_paths:
                    st.error("Please provide at least one file.")
                else:
                    pdf_paths = [p for p in new_paths if p.lower().endswith(".pdf")]
                    docx_paths = [p for p in new_paths if p.lower().endswith(".docx")]
                    pages_data = []
                    if pdf_paths:
                        with st.spinner(f"Extracting text from {len(pdf_paths)} PDF(s)…"):
                            pages_data += process_pdf_files(pdf_paths)
                    if docx_paths:
                        with st.spinner(f"Extracting text from {len(docx_paths)} DOCX file(s)…"):
                            pages_data += process_docx_files(docx_paths)
                    with st.spinner("Chunking and embedding new files…"):
                        new_chunks = get_chunks(pages_data, st.session_state["chunk_size"], st.session_state["overlap"])
                        new_vectors = create_vector_store(new_chunks)
                    st.session_state["database"].extend(new_vectors)
                    st.session_state["all_paths"].extend(new_paths)
                    st.session_state["n_chunks"] += len(new_chunks)
                    st.session_state["n_pages"] += len(pages_data)
                    st.success(f"Added {len(new_chunks)} chunks from {len(pages_data)} page(s).")
                    st.rerun()

        # ── Restart ──────────────────────────────────────────────────────────
        if st.button("🗑️ Restart — Clear Knowledge Base", type="secondary"):
            for key in ["database", "all_paths", "chunk_size", "overlap", "k_value", "chat_history", "n_chunks", "n_pages"]:
                st.session_state.pop(key, None)
            st.rerun()

        st.divider()

        # ── Chat history ─────────────────────────────────────────────────────
        for msg in st.session_state.get("chat_history", []):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and "sources" in msg:
                    st.caption(f"Sources: {msg['sources']}")

        # ── Chat input ───────────────────────────────────────────────────────
        user_query = st.chat_input("Ask a question about your documents…")
        if user_query:
            with st.chat_message("user"):
                st.markdown(user_query)
            st.session_state["chat_history"].append({"role": "user", "content": user_query})

            with st.spinner("Searching…"):
                relevant_chunks = retrieve_k_relevant_chunks(
                    user_query, st.session_state["database"], st.session_state["k_value"]
                )
                context = "\n\n---\n\n".join(c["text"] for c in relevant_chunks)
                answer = generate_answer(user_query, context)

            sources = ", ".join(f"{c['source_file']} (p. {c['page']})" for c in relevant_chunks)

            with st.chat_message("assistant"):
                st.markdown(answer)
                st.caption(f"Sources: {sources}")

            st.session_state["chat_history"].append(
                {"role": "assistant", "content": answer, "sources": sources}
            )

            if answer.lower().strip() == "i don't have enough data":
                st.warning(
                    "Not enough context found. Try increasing **K** or adjusting **Chunk Size** "
                    "in the ⚙️ Settings panel above."
                )


# ─── Mode 2: Batch Q&A ────────────────────────────────────────────────────────

elif mode == "📋 Batch Q&A":
    st.header("Batch Q&A Processor")
    st.markdown("Provide a file with one question per line (.txt, .docx, or .pdf).")

    uploaded = st.file_uploader(
        "Drag & drop your questions file here",
        type=["txt", "docx", "pdf"],
        key="batch_upload",
    )
    path_input = st.text_input(
        "Or paste the file path",
        placeholder="/path/to/questions.txt",
        key="batch_path",
    )

    if st.button("Process Questions", type="primary"):
        # Resolve the source file
        tmp_path = None
        source_path = None

        if uploaded:
            tmp_path = save_uploaded_file(uploaded)
            source_path = tmp_path
        elif path_input.strip():
            raw = path_input.strip().strip('"').strip("'")
            if not os.path.isfile(raw):
                st.error(f"File not found: {raw}")
                st.stop()
            source_path = raw
        else:
            st.error("Please upload a file or enter a file path.")
            st.stop()

        try:
            question_list = read_questions_from_file(source_path)
            if not question_list:
                st.error("No questions found in the file.")
                st.stop()

            answer_list = []
            progress = st.progress(0, text="Answering questions…")

            for i, question in enumerate(question_list):
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": question},
                    ],
                    temperature=0.7,
                )
                answer_list.append((question, response.choices[0].message.content))
                progress.progress((i + 1) / len(question_list), text=f"Question {i + 1}/{len(question_list)}")

            st.success(f"Done! Answered {len(answer_list)} question(s).")

            with st.expander("View Answers", expanded=True):
                for q, a in answer_list:
                    st.markdown(f"**Q: {q}**")
                    st.markdown(a)
                    st.divider()

            output_text = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in answer_list)
            st.download_button(
                label="⬇️ Download answers.txt",
                data=output_text,
                file_name="answers.txt",
                mime="text/plain",
            )

        except ValueError as e:
            st.error(str(e))
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
