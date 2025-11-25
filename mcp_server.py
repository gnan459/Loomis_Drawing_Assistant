import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import cv2
import os
from pose_detection import detect_pose_and_face
from geometry_utils import calculate_head_dimensions, compute_face_turn_angle
from render_steps import *

# Create server instance
server = Server("loomis-drawing-assistant")

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="detect_face",
            description="Detect face landmarks and analyze head geometry from an image",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "Path to the image file"}
                },
                "required": ["image_path"]
            }
        ),
        Tool(
            name="apply_loomis",
            description="Apply Loomis method construction lines to an image and save the result",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "Path to the input image"},
                    "output_path": {"type": "string", "description": "Path where output should be saved"}
                },
                "required": ["image_path", "output_path"]
            }
        ),
        Tool(
            name="analyze_proportions",
            description="Analyze facial proportions and provide measurements",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "Path to the image file"}
                },
                "required": ["image_path"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    
    if name == "detect_face":
        image_path = arguments["image_path"]
        
        if not os.path.exists(image_path):
            return [TextContent(type="text", text=f"Error: Image not found at {image_path}")]
        
        data = detect_pose_and_face(image_path, display=False)
        if not data['face']:
            return [TextContent(type="text", text="No face detected in image")]
        
        radius, center, _ = calculate_head_dimensions(data['face'])
        direction = compute_face_turn_angle(data['face'])
        
        result = f"Face detected!\nCenter: {center}\nRadius: {radius}px\nDirection: {direction}"
        return [TextContent(type="text", text=result)]
    
    elif name == "apply_loomis":
        image_path = arguments["image_path"]
        output_path = arguments["output_path"]
        
        if not os.path.exists(image_path):
            return [TextContent(type="text", text=f"Error: Image not found at {image_path}")]
        
        data = detect_pose_and_face(image_path, display=False)
        if not data['face']:
            return [TextContent(type="text", text="No face detected in image")]
        
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
        result = f"Loomis construction saved to {output_path}"
        return [TextContent(type="text", text=result)]
    
    elif name == "analyze_proportions":
        image_path = arguments["image_path"]
        
        if not os.path.exists(image_path):
            return [TextContent(type="text", text=f"Error: Image not found at {image_path}")]
        
        data = detect_pose_and_face(image_path, display=False)
        if not data['face']:
            return [TextContent(type="text", text="No face detected in image")]
        
        radius, center, _ = calculate_head_dimensions(data['face'])
        direction = compute_face_turn_angle(data['face'])
        
        xs = [p[0] for p in data['face']]
        ys = [p[1] for p in data['face']]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        
        result = f"""Facial Proportions:
Head radius: {radius}px
Center: {center}
Face: {width:.0f}x{height:.0f}px
Ratio: {width/height:.2f}
Direction: {direction}"""
        return [TextContent(type="text", text=result)]
    
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
