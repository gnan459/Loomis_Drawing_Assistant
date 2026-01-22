from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import cv2
import os
from pose_detection import detect_pose_and_face
from geometry_utils import calculate_head_dimensions, compute_face_turn_angle
from render_steps import *
from drawing_instructor_agent import generate_drawing_instructions

from qdrant_setup import init_qdrant
from embedding_utils import geometry_to_vector
from memory_store import store_in_qdrant
from memory_retriever import retrieve_similar


load_dotenv()

qdrant_client = init_qdrant()


@tool
def detect_face(image_path: str) -> str:
    """Detect face landmarks and analyze head geometry from an image."""
    if not os.path.exists(image_path):
        return f"Error: Image not found at {image_path}"
    
    data = detect_pose_and_face(image_path, display=False)
    if not data['face']:
        return "No face detected in image"
    
    radius, center, _ = calculate_head_dimensions(data['face'])
    direction = compute_face_turn_angle(data['face'])
    
    return f"Face detected!\nCenter: {center}\nRadius: {radius}px\nDirection: {direction}"

@tool
def apply_loomis(image_path: str, output_path: str) -> str:
    """Apply Loomis method construction lines to an image and save the result."""
    if not os.path.exists(image_path):
        return f"Error: Image not found at {image_path}"
    
    data = detect_pose_and_face(image_path, display=False)
    if not data['face']:
        return "No face detected in image"
    
    radius, center, _ = calculate_head_dimensions(data['face'])
    direction = compute_face_turn_angle(data['face'])

    geometry = {
    "radius": radius,
    "center_x": center[0],
    "center_y": center[1],
    "direction": 1 if direction == "left" else -1
    }

    vector = geometry_to_vector(geometry)

    store_in_qdrant(
        qdrant_client,
        vector,
        payload={
            "type": "reference",
            "image_path": image_path
        }
    )

    
    img = cv2.imread(image_path)
    img = construct_loomis_sphere(img, center, radius, direction, data["face"])
    img = construct_vertical_line(img, data["face"])
    img = construct_brow_line(img, data["face"])
    img = construct_nose_line(img, data["face"])
    img = construct_chin_line(img, data["face"])
    img = construct_ellipse_vertical_line(img, center, radius, direction, data["face"])
    img = construct_jaw_line(img, data["face"])
    img = construct_outer_face_line(img, data["face"], direction)
    
    cv2.imwrite(output_path, img)
    return f"Loomis construction saved to {output_path}"

@tool
def analyze_proportions(image_path: str) -> str:
    """Analyze facial proportions and provide measurements."""
    if not os.path.exists(image_path):
        return f"Error: Image not found at {image_path}"
    
    data = detect_pose_and_face(image_path, display=False)
    if not data['face']:
        return "No face detected in image"
    
    radius, center, _ = calculate_head_dimensions(data['face'])
    direction = compute_face_turn_angle(data['face'])
    
    xs = [p[0] for p in data['face']]
    ys = [p[1] for p in data['face']]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    
    return f"""Facial Proportions:
Head radius: {radius}px
Center: {center}
Face: {width:.0f}x{height:.0f}px
Ratio: {width/height:.2f}
Direction: {direction}"""

@tool
def explain_loomis_guidelines(image_path: str) -> str:
    """Generate a step-by-step Loomis drawing tutorial based on detected guidelines."""
    
    if not os.path.exists(image_path):
        return "Error: Image not found."

    data = detect_pose_and_face(image_path, display=False)
    if not data["face"]:
        return "No face detected."

    radius, center, _ = calculate_head_dimensions(data["face"])
    direction = compute_face_turn_angle(data["face"])

    geometry = {
    "radius": radius,
    "center_x": center[0],
    "center_y": center[1],
    "direction": 1 if direction == "left" else -1
    }

    vector = geometry_to_vector(geometry)

    store_in_qdrant(
        qdrant_client,
        vector,
        payload={
            "type": "user_drawing",
            "image_path": image_path
        }
    )


    xs = [p[0] for p in data["face"]]
    ys = [p[1] for p in data["face"]]
    proportions = {
        "width": max(xs) - min(xs),
        "height": max(ys) - min(ys),
        "ratio": (max(xs) - min(xs)) / (max(ys) - min(ys)),
    }

    similar_examples = retrieve_similar(qdrant_client, vector)


    tutorial = generate_drawing_instructions(
        direction=str(direction),
        radius=radius,
        center=center,
        proportions=proportions,
        notes=f"""
Based on {len(similar_examples)} similar past Loomis constructions
retrieved from visual memory with similar head orientation and proportions.
"""


    )

    return tutorial

# Create tools list
TOOLS = [detect_face, apply_loomis, analyze_proportions, explain_loomis_guidelines]

# System message
SYSTEM_MESSAGE = """You're a Loomis method drawing assistant. Help users apply geometric construction to draw heads accurately." \
"
TOOL ROUTING RULES:
- When the user says: "Explain guidelines for <path>" or similar,
  ALWAYS call the `explain_loomis_guidelines` tool.
- NEVER answer guideline explanations yourself.
- ONLY the tool should generate the step-by-step tutorial.

You MUST call a tool for every request.
You are not allowed to answer on your own.

Your job is to choose the correct tool and call it."""

# Create agent
llm = ChatOllama(model="qwen2.5:1.5b", temperature=0)
agent = create_react_agent(llm, TOOLS, prompt=SYSTEM_MESSAGE)

def run_agent(user_input: str) -> str:
    
    """
    Hybrid tool-or-LLM routing.
    - If the message matches a known tool pattern → call tool directly.
    - Otherwise → send to LLM via REACT.
    """

    text = user_input.lower().strip()

    # -------------------------------
    # 1. EXPLAIN GUIDELINES
    # -------------------------------
    if text.startswith("explain guidelines for"):
        try:
            path = user_input.split("for", 1)[1].strip()
            return explain_loomis_guidelines.run({"image_path": path})
        except Exception as e:
            return f"Error running explain_guidelines: {str(e)}"


    # -------------------------------
    # 2. APPLY LOOMIS METHOD
    # -------------------------------
    if text.startswith("apply loomis to"):
        try:
            # Example:
            # "Apply Loomis to A and save to B"
            parts = user_input.split("to", 1)[1].strip()
            img_path, out_path = parts.split("and save to")
            img_path = img_path.strip()
            out_path = out_path.replace("save to", "").strip()

            return apply_loomis.run({
                "image_path": img_path,
                "output_path": out_path
            })
        except Exception as e:
            return f"Error running apply_loomis: {str(e)}"


    # -------------------------------
    # 3. DETECT FACE
    # -------------------------------
    if text.startswith("detect face in"):
        try:
            path = user_input.split("in", 1)[1].strip()
            return detect_face.run({"image_path": path})
        except Exception as e:
            return f"Error running detect_face: {str(e)}"


    # -------------------------------
    # 4. ANALYZE PROPORTIONS
    # -------------------------------
    if text.startswith("analyze proportions in"):
        try:
            path = user_input.split("in", 1)[1].strip()
            return analyze_proportions.run({"image_path": path})
        except Exception as e:
            return f"Error running analyze_proportions: {str(e)}"


    # -------------------------------
    # 5. OTHERWISE → NORMAL LLM (REACT)
    # -------------------------------
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config={"recursion_limit": 50}
        )
        return result["messages"][-1].content

    except Exception as e:
        return f"Error: {str(e)}"


def main():
    """Interactive CLI."""
    print("🎨 Loomis Drawing Assistant")
    print("=" * 50)
    print("Available commands:")
    print("  - Detect face in <image_path>")
    print("  - Apply Loomis to <image_path> and save to <output_path>")
    print("  - Analyze proportions in <image_path>")
    print("  - Explain guidelines for <image_path>")
    print("  - Type 'exit' or 'quit' to stop")
    print("=" * 50)
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye! 👋")
            break
        
        if not user_input:
            continue
            
        response = run_agent(user_input)
        print(f"\nAssistant: {response}")

if __name__ == "__main__":
    main()
