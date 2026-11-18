
import ollama
import time

MODEL = "qwen3:4b-instruct"


def ask_ai(question):
    print("\nOpenResearch AI is thinking...\n")

    start = time.time()

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are OpenResearch AI, an academic "
                        "research assistant. Give clear, concise, "
                        "structured and evidence-conscious answers. "
                        "Never invent research citations."
                    )
                },
                {
                    "role": "user",
                    "content": question + "\n/no_think"
                }
            ],
            options={
                "temperature": 0.3,
                "num_ctx": 2048,
                "num_predict": 350
            }
        )

        answer = response["message"]["content"]

        print("=" * 60)
        print("OPENRESEARCH AI")
        print("=" * 60)
        print(answer)
        print("=" * 60)

        elapsed = time.time() - start
        print(f"Response time: {elapsed:.2f} seconds")

    except Exception as error:
        print("\nSomething went wrong:")
        print(error)


if __name__ == "__main__":
    print("=" * 60)
    print("WELCOME TO OPENRESEARCH AI")
    print("=" * 60)

    question = input("\nEnter your research question: ").strip()

    if question:
        ask_ai(question)
    else:
        print("Please enter a valid research question.")
