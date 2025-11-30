# Loomis Drawing Assistant  
AI-powered geometric construction system for the Loomis head-drawing method

## 1. Overview
Constructing a realistic human head requires accurate proportions. The Loomis Method breaks the head into simple geometric forms, but applying these guidelines manually from a reference image is difficult and slow. This project automates the Loomis construction process by detecting facial landmarks, computing precise geometric measurements, and generating guiding lines directly on the image.

## 2. Problem Statement
Beginners struggle with:
- Understanding the proportions of the human head
- Determining the head orientation
- Manually drawing the Loomis sphere and face planes
- Accurately placing brow, nose, chin, and jaw guidelines
- Keeping drawings consistent across references

## 3. Solution Summary
The Loomis Drawing Assistant:
- Detects facial landmarks via MediaPipe Face Mesh
- Computes head radius, center, and orientation
- Determines left/right/forward head direction
- Draws Loomis sphere, side-plane ellipse, vertical axis, brow line, nose line, chin line, jawline, and outer face contour
- Uses a ReAct-based LLM agent to analyze images and call tools
- Exposes tools through an MCP server
- Runs entirely locally through Ollama without cloud dependencies

## 4. Architecture
User Input → LLM Agent → Tools → MediaPipe → Geometry Utils → Render Steps → Output

Core components:
- agent.py
- mcp_server.py
- pose_detection.py
- geometry_utils.py
- render_steps.py

## 5. Features Demonstrated
- Multi-agent tool workflow
- Custom tools
- MCP integration
- Local LLM reasoning via Ollama
- Context engineering
- MediaPipe CV pipeline

## 6. Installation & Setup

Clone the repository:
```
git clone https://github.com/yourname/Loomis_Drawing_Assistant.git
cd Loomis_Drawing_Assistant
```

Create virtual environment:
```
python -m venv venv
```

Install dependencies:
```
pip install -r requirements.txt
```

Install Ollama and pull the model:
```
ollama pull qwen2.5:1.5b
```

Run the agent:
```
python agent.py
```

## 7. Usage Examples

Detect face:
```
Detect face in D:/images/photo.jpg
```

Apply Loomis:
```
Apply Loomis to D:/images/photo.jpg and save to D:/output/loomis.jpg
```

## 8. Example Output Description
Output includes sphere, ellipse, centerline, brow line, nose line, chin line, jawline, and outer contour.

## 9. Value Proposition
The tool automates Loomis construction, helping artists produce consistent proportions.

## 11. Conclusion
A complete AI pipeline integrating CV, geometry, LLM reasoning, and tool orchestration to generate Loomis construction guides.
