import os
from openai import OpenAI
from dotenv import load_dotenv

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

        with open('answers.text', 'w') as answer_file:
            answer_file.write("\n\n".join(answer_list))
        print("\n[Success] Batch processing complete. Check 'answers.text'.")
    except FileNotFoundError:
        print(f"\n[Error] The file '{question_file}' was not found.")


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
            chunks = get_chunks('knowledge.text', chunk_size, overlap)
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
        new_answer_file_to_a_general_question_file('question.text')

    else:
        print("\n[System] Invalid choice. Program terminated.")


if __name__ == "__main__":
    main()