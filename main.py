import os
from openai import OpenAI
from dotenv import load_dotenv
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


def new_answer_file_to_a_general_question_file(question_file):
    """Processes a batch of questions from a file and saves answers to answers.text."""
    try:
        with open(question_file, 'r') as file:
            question_list = [line.strip() for line in file if line.strip()]

        answer_list = []
        for question in question_list:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question}
                ],
                temperature=0.7
            )
            answer = response.choices[0].message.content
            answer_list.append(f"Q: {question}\nA: {answer}")

        with open('answers.txt', 'w') as answer_file:
            answer_file.write("\n\n".join(answer_list))
        print("\n[Success] Batch processing complete. Check 'answers.txt'.")
    except FileNotFoundError:
        print(f"\n[Error] The file '{question_file}' was not found.")


SCANNED_TEXT_THRESHOLD = 20  # Min characters per page to consider a PDF text-based


def is_scanned_pdf(pdf_path):
    """Returns True if the PDF is scanned (image-based), False if it contains real text."""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text and len(text.strip()) > SCANNED_TEXT_THRESHOLD:
                return False
    return True


def extract_table_as_text(table):
    """Converts a pdfplumber table (list of rows) into a readable string."""
    if not table:
        return ""
    rows = []
    for row in table:
        row_cells = [str(cell).strip() if cell is not None else "" for cell in row]
        rows.append(" | ".join(row_cells))
    return "\n".join(rows)


def extract_image_text_from_page(page):
    """Runs OCR on images embedded inside a text-based PDF page (Hebrew + English)."""
    image_text_parts = []
    for img_obj in page.images:
        try:
            page_image = page.to_image(resolution=200).original
            bbox = (img_obj['x0'], img_obj['top'], img_obj['x1'], img_obj['bottom'])
            cropped = page_image.crop(bbox)
            ocr_text = pytesseract.image_to_string(cropped, lang='heb+eng').strip()
            if ocr_text:
                image_text_parts.append(ocr_text)
        except Exception:
            continue
    return "\n".join(image_text_parts)


def extract_text_from_text_pdf(pdf_path):
    """Extracts text, tables, and image-embedded text from a text-based PDF, page by page."""
    pages_data = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            parts = []

            # Regular text
            page_text = page.extract_text()
            if page_text and page_text.strip():
                parts.append(page_text.strip())

            # Tables
            tables = page.extract_tables()
            for table in tables:
                table_text = extract_table_as_text(table)
                if table_text:
                    parts.append(f"[TABLE]\n{table_text}\n[END TABLE]")

            # Text inside embedded images (OCR)
            if page.images:
                image_text = extract_image_text_from_page(page)
                if image_text:
                    parts.append(f"[IMAGE TEXT]\n{image_text}\n[END IMAGE TEXT]")

            combined_text = "\n\n".join(parts)
            if combined_text.strip():
                pages_data.append({
                    "text": combined_text.strip(),
                    "source_file": os.path.basename(pdf_path),
                    "page": page_num
                })
    return pages_data


def extract_text_from_scanned_pdf(pdf_path):
    """Uses OCR to extract text from all pages of a scanned PDF (supports Hebrew + English)."""
    pages_data = []
    images = convert_from_path(pdf_path, dpi=300)
    for page_num, image in enumerate(images, start=1):
        ocr_text = pytesseract.image_to_string(image, lang='heb+eng').strip()
        if ocr_text:
            pages_data.append({
                "text": ocr_text,
                "source_file": os.path.basename(pdf_path),
                "page": page_num
            })
    return pages_data


def extract_text_from_pdf(pdf_path):
    """Detects if a PDF is scanned or text-based and extracts its content accordingly.

    Returns a list of page dicts: [{"text": ..., "source_file": ..., "page": ...}, ...]
    Each dict represents one page and carries the metadata needed for future chunk referencing.
    """
    file_name = os.path.basename(pdf_path)
    if is_scanned_pdf(pdf_path):
        print(f"[INFO] '{file_name}' is a scanned PDF — using OCR.")
        return extract_text_from_scanned_pdf(pdf_path)
    else:
        print(f"[INFO] '{file_name}' is a text-based PDF — extracting directly.")
        return extract_text_from_text_pdf(pdf_path)


def process_pdf_files(pdf_paths):
    """Processes a list of PDF file paths and returns combined page-level extracted data.

    Each entry contains the text, source file name, and page number so that
    when the text is later split into chunks, every chunk can reference its origin.
    """
    all_pages = []
    for pdf_path in pdf_paths:
        print(f"\n[Processing] {os.path.basename(pdf_path)}")
        pages = extract_text_from_pdf(pdf_path)
        all_pages.extend(pages)
        print(f"[Done] Extracted {len(pages)} page(s) from '{os.path.basename(pdf_path)}'")
    print(f"\n[Summary] Total pages extracted: {len(all_pages)}")
    return all_pages












def get_chunks(file_path, chunk_size, overlap):
    """Splits the knowledge base into overlapping segments."""
    with open(file_path, 'r', encoding='utf-8') as file:
        full_text = file.read()

    chunk_list = []
    start = 0
    while start < len(full_text):
        chunk = full_text[start: start + chunk_size]
        chunk_list.append(chunk)
        start = start + chunk_size - overlap
    return chunk_list


def create_vector_store(chunks_list):
    """Generates embeddings for all text chunks."""
    response = client.embeddings.create(
        input=chunks_list,
        model="text-embedding-3-small"
    )

    vector_store = []
    for i, item in enumerate(response.data):
        vector_store.append({
            "text": chunks_list[i],
            "vector": item.embedding
        })
    return vector_store


def retrieve_k_relevant_chunks(user_query, database, k_value):
    """Finds top-K most relevant chunks based on vector similarity."""
    response = client.embeddings.create(
        input=user_query,
        model="text-embedding-3-small"
    )
    query_vector = response.data[0].embedding

    results = []
    for item in database:
        # Calculate dot product similarity
        score = sum(a * b for a, b in zip(query_vector, item["vector"]))
        results.append({"text": item["text"], "score": score})

    # Sort results by similarity score
    results.sort(key=lambda x: x["score"], reverse=True)
    return [item["text"] for item in results[:k_value]]


def generate_answer(user_query, context):
    """Sends the retrieved context to GPT-4o for a response."""
    prompt = f"""
    Answer the following question using ONLY the provided context.
    If the answer is not in the context, your response MUST be exactly: "i don't have enough data"

    Context:
    {context}

    Question:
    {user_query}
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that strictly follows the provided context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content


def main():
    print("--- Welcome to the Advanced RAG & Q&A Tool ---")
    print("\nAvailable Modes:")
    print("1. Interactive RAG Chat")
    print("2. Batch Q&A Processor")

    user_choice = input("\nChoose mode (1 or 2): ").strip()
    if user_choice == "1":
        # --- RAG Mode Execution ---
        try:
            print("\n--- RAG Setup ---")
            c_size = input("Enter Chunk Size [default 500]: ") or "500"
            c_overlap = input("Enter Overlap [default 50]: ") or "50"

            chunk_size = int(c_size.strip())
            overlap = int(c_overlap.strip())
            k_value = 1  # Starting with K=1 to demonstrate the system's ability to adjust

            print(f"\nInitializing RAG (Size: {chunk_size}, Overlap: {overlap})...")
            chunks = get_chunks('knowledge.txt', chunk_size, overlap)
            database = create_vector_store(chunks)
            print("\n[System] RAG Mode Activated. Type 'exit' to quit.")

            while True:
                user_query = input("\nUser: ")
                if user_query.lower() == 'exit':
                    break
                if not user_query.strip():
                    continue

                # Retrieval Step
                relevant_chunks = retrieve_k_relevant_chunks(user_query, database, k_value)
                context = "\n\n---\n\n".join(relevant_chunks)

                # Generation Step
                answer = generate_answer(user_query, context)
                print(f"\nAI (K={k_value}): {answer}")

                # Smart Logic: Offer to increase K if context is insufficient
                if answer.lower() == "i don't have enough data":
                    print("\n[System Alert] The information was not found in the current context.")
                    suggestion = input("Would you like to try again with a larger K-value? (yes/no): ")

                    if suggestion.lower().strip() == "yes":
                        try:
                            new_k = input("Enter new K-value (e.g., 3 or 5): ")
                            k_value = int(new_k)
                            print(f"[System] K-value updated to {k_value}. Please ask your question again.")
                        except ValueError:
                            print("[Error] Invalid number. Keeping current K.")
                    else:
                        exit_confirm = input("Would you like to exit the program? (yes/no): ")
                        if exit_confirm.lower().strip() == "yes":
                            break

                print("-" * 50)

        except Exception as e:
            print(f"\n[Error] An error occurred during setup: {e}")

    elif user_choice == "2":
        # --- Batch Mode Execution ---
        print("\n--- Batch Mode Activated ---")
        new_answer_file_to_a_general_question_file('question.txt')

    else:
        print("\n[System] Invalid choice. Program terminated.")


if __name__ == "__main__":
    main()

