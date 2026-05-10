import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
_client = None


def client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def get_chunks(file_path, chunk_size, overlap):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


def create_vector_store(chunks):
    response = client().embeddings.create(input=chunks, model="text-embedding-3-small")
    return [{"text": chunks[i], "vector": item.embedding} for i, item in enumerate(response.data)]


def retrieve_k_relevant_chunks(query, database, k):
    response = client().embeddings.create(input=query, model="text-embedding-3-small")
    query_vector = response.data[0].embedding

    scored = [
        {"text": item["text"], "score": sum(a * b for a, b in zip(query_vector, item["vector"]))}
        for item in database
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return [item["text"] for item in scored[:k]]


def generate_answer(query, context):
    prompt = f"""Answer the following question using ONLY the provided context.
If the answer is not in the context, respond with exactly: "i don't have enough data"

Context:
{context}

Question:
{query}"""

    response = client().chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You answer strictly from the provided context."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


def batch_process_questions(question_file, knowledge_file, chunk_size=500, overlap=50, k=3):
    """Run each question in question_file through the full RAG pipeline and write answers.txt."""
    try:
        with open(question_file, 'r') as f:
            questions = [line.strip() for line in f if line.strip()]

        print(f"\nBuilding vector store from '{knowledge_file}'...")
        chunks = get_chunks(knowledge_file, chunk_size, overlap)
        database = create_vector_store(chunks)
        print(f"[System] Ready. Processing {len(questions)} questions with K={k}...")

        answers = []
        for i, question in enumerate(questions, 1):
            print(f"  [{i}/{len(questions)}] {question[:70]}...")
            relevant_chunks = retrieve_k_relevant_chunks(question, database, k)
            context = "\n\n---\n\n".join(relevant_chunks)
            answer = generate_answer(question, context)
            answers.append(f"Q: {question}\nA: {answer}")

        with open('answers.txt', 'w') as f:
            f.write("\n\n".join(answers))
        print("\n[Success] Batch complete. Check 'answers.txt'.")

    except FileNotFoundError as e:
        print(f"\n[Error] File not found: {e}")


def main():
    print("--- RAG from Scratch ---")
    print("\n1. Interactive RAG Chat")
    print("2. Batch Q&A Processor")
    choice = input("\nChoose mode (1 or 2): ").strip()

    if choice == "1":
        try:
            chunk_size = int(input("Chunk size [default 500]: ") or 500)
            overlap    = int(input("Overlap    [default  50]: ") or 50)
            k = 1

            print(f"\nBuilding vector store (chunk={chunk_size}, overlap={overlap})...")
            chunks   = get_chunks('knowledge.txt', chunk_size, overlap)
            database = create_vector_store(chunks)
            print("[System] Ready. Type 'exit' to quit.\n")

            while True:
                query = input("You: ").strip()
                if query.lower() == 'exit':
                    break
                if not query:
                    continue

                relevant = retrieve_k_relevant_chunks(query, database, k)
                context  = "\n\n---\n\n".join(relevant)
                answer   = generate_answer(query, context)
                print(f"\nAI (K={k}): {answer}\n" + "-" * 50)

                if answer.strip().lower() == "i don't have enough data":
                    print("[System] Context was insufficient.")
                    if input("Try with a larger K? (yes/no): ").strip().lower() == "yes":
                        try:
                            k = int(input("New K value: "))
                            print(f"[System] K updated to {k}. Ask your question again.")
                        except ValueError:
                            print("[Error] Invalid number. Keeping current K.")
                    elif input("Exit? (yes/no): ").strip().lower() == "yes":
                        break

        except Exception as e:
            print(f"\n[Error] {e}")

    elif choice == "2":
        batch_process_questions('question.txt', 'knowledge.txt')

    else:
        print("[System] Invalid choice.")


if __name__ == "__main__":
    main()
