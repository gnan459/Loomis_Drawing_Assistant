from langchain_ollama import ChatOllama

instructor_llm = ChatOllama(
    model = "qwen2.5:1.5b",
    temperature=0.5
)

def generate_drawing_instructions(
    direction: str,
    radius: float,
    center: tuple,
    proportions: dict,
    notes: str = ""
) -> str:
    cx, cy = center

    prompt = f"""
You are a professional art teacher trained in the Loomis head construction method.

The system has already detected:
- Head direction: {direction}
- Head radius: {radius:.1f}
- Head center: {center}
- Facial proportions: {proportions}

Retrieved visual memory context:
{notes}

The following Loomis guidelines were auto-generated:
1. Cranial circle (base Loomis sphere)
2. Side-plane ellipse (showing head direction)
3. Vertical centerline
4. Brow line
5. Nose guideline
6. Chin guideline
7. Jawline curve
8. Outer face contour

Write a detailed but simple drawing tutorial for a beginner artist.
...
"""
    response = instructor_llm.invoke(prompt)
    return response.content

