from backend.llm import generate_response

if __name__ == "__main__":
    print("Testing Gemini connection...\n")

    response = generate_response(
        "What is CellScan AI, and which model architecture does it use?"
    )

    print("Gemini Response:")
    print(response)