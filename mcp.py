from fastmcp import FastMCP
import cv2
import os
from pose_detection import detect_pose_and_face
from geometry_utils import calculate_head_dimensions, compute_face_turn_angle
from render_steps import *

# Create FastMCP server
mcp = FastMCP("Loomis Drawing Assistant")

@mcp.tool()
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

@mcp.tool()
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

@mcp.tool()
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

if __name__ == "__main__":
    mcp.run()
