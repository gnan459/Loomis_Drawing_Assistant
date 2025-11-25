from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import cv2
import os
from pose_detection import detect_pose_and_face
from geometry_utils import calculate_head_dimensions, compute_face_turn_angle
from render_steps import *

load_dotenv()

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

# Create tools list
TOOLS = [detect_face, apply_loomis, analyze_proportions]

# System message
SYSTEM_MESSAGE = "You're a Loomis method drawing assistant. Help users apply geometric construction to draw heads accurately."

# Create agent
llm = ChatOllama(model="qwen2.5:1.5b", temperature=0)
agent = create_react_agent(llm, TOOLS, prompt=SYSTEM_MESSAGE)

def run_agent(user_input: str) -> str:
    """Run the agent."""
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
