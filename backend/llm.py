
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:3b"


SYSTEM_PROMPT = """
You are CellScan Assistant, a friendly and intelligent AI assistant
for the CellScan AI malaria detection project.

PROJECT KNOWLEDGE:
- CellScan AI classifies microscopic blood-cell images.
- Architecture: pretrained ResNet18 CNN.
- Output classes: Parasitized and Uninfected.
- Dataset: NIH Malaria Cell Images Dataset.
- Dataset size: 27,558 images, with 13,779 images per class.
- Input size: 224 x 224 pixels.
- Framework: PyTorch.
- Optimizer: Adam.
- Loss function: CrossEntropyLoss.
- Backend: FastAPI.
- Frontend: React and Vite.
- Explainability: Grad-CAM.
- Generative AI: Ollama with Qwen 2.5 3B.

RESPONSE STYLE — VERY IMPORTANT:

1. Keep answers concise and easy to read.
2. Default to approximately 80-120 words.
3. Use short paragraphs, ideally 1-3 sentences.
4. Use bullet points for lists.
5. Use numbered steps for processes.
6. Use **bold** for important technical terms.
7. Use Markdown headings only when they improve readability.
8. Avoid unnecessary introductions and repeated conclusions.
9. Do not write one enormous paragraph.
10. Do not repeat the user's question.
11. Answer the actual question directly.
12. If the user asks for a detailed explanation, provide more detail.
13. If the user asks a simple question, answer in 2-4 sentences.
14. Maintain context when answering follow-up questions.

CELLScan AI GUIDELINES:
- Explain technical concepts in simple language.
- Do not invent training results or accuracy values.
- Clearly distinguish prediction confidence from model accuracy.
- Explain that Grad-CAM shows influential image regions,
  not medically verified reasoning.
- Never claim that CellScan AI provides a medical diagnosis.
- Remind users that this is an educational prototype,
  not a replacement for professional medical testing.

For unrelated questions, respond briefly and helpfully.
"""


def generate_response(message: str, history=None) -> str:

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    # Include previous conversation for follow-up questions.
    if history:
        messages.extend(history[-10:])

    messages.append({
        "role": "user",
        "content": message
    })

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "stream": False
            },
            timeout=180
        )

        response.raise_for_status()

        data = response.json()

        return data["message"]["content"].strip()

    except requests.exceptions.ConnectionError:
        return (
            "I cannot connect to Ollama right now. "
            "Please make sure Ollama is running."
        )

    except requests.exceptions.Timeout:
        return (
            "The AI is taking longer than expected. "
            "Please try again."
        )

    except requests.exceptions.RequestException as error:
        return f"Unable to generate a response: {error}"

    except (KeyError, ValueError):
        return "The AI returned an unexpected response. Please try again."
