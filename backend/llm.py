
import os
from google import genai
from dotenv import load_dotenv
load_dotenv()
MODEL_NAME = "gemini-3.5-flash-lite"

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
- Generative AI: Google Gemini API.

RESPONSE STYLE:
1. Keep answers concise and easy to read.
2. Default to approximately 80-120 words.
3. Use short paragraphs and bullet points.
4. Use numbered steps for processes.
5. Use **bold** for important technical concepts.
6. Use Markdown where helpful.
7. Avoid unnecessary introductions.
8. Answer the actual question directly.
9. Provide more detail when explicitly requested.
10. Maintain context in follow-up questions.

CELLSCAN AI GUIDELINES:
- Explain technical concepts in simple language.
- Do not invent training results or accuracy values.
- Distinguish prediction confidence from model accuracy.
- Explain that Grad-CAM shows influential image regions,
  not medically verified reasoning.
- Never claim CellScan AI provides a medical diagnosis.
- Remind users this is an educational prototype,
  not a replacement for professional medical testing.

For unrelated questions, respond briefly and helpfully.
"""

def generate_response(message: str, history=None) -> str:

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return "The AI assistant is not configured. Please contact the administrator."

    try:
        client = genai.Client(api_key=api_key)

        conversation = []

        if history:
            for item in history[-10:]:
                role = item.get("role")
                content = item.get("content", "")

                if role in ["user", "assistant"] and content:
                    conversation.append(f"{role}: {content}")

        conversation.append(f"user: {message}")

        prompt = (
            SYSTEM_PROMPT
            + "\n\nCONVERSATION HISTORY:\n"
            + "\n".join(conversation)
            + "\n\nassistant:"
        )

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        if not response.text:
            return "The AI returned an empty response. Please try again."

        return response.text.strip()

    except Exception as error:
        print(f"Gemini API Error: {error}")
        return "Unable to connect to the AI service right now. Please try again later."