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
    """
    Agent 2: Drawing Improvement Instructor.
    Turns Loomis guidelines into a beginner-friendly step-by-step drawing guide.
    """
    cx, cy = center

    prompt = f"""
You are a professional art teacher trained in the Loomis head construction method.

The system has already detected:
- Head direction: {direction}
- Head radius: {radius:.1f}
- Head center: {center}
- Facial proportions: {proportions}

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
The tutorial must:

- Use short steps (Step 1, Step 2, ...)
- Describe how to draw the head from scratch on paper using these guides
- Explain why each step matters
- Include tips for symmetry, perspective, and proportion
- Avoid talking about pixels or code
- Only explain drawing, not technical implementation

End with a short section titled "Common Beginner Mistakes" (3–5 bullets).
"""
    response = instructor_llm.invoke(prompt)
    return response.content
