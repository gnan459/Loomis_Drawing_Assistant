Loomis Drawing Assistant
AI-powered construction-line generator for the Loomis head-drawing method
1. Problem Statement

Constructing a human head accurately is a complex task for both beginner and intermediate artists. The Loomis Method is a widely used technique that breaks the head into simplified geometric forms—sphere, centerline, jawline, brow line, and proportional divisions. However, applying these guidelines manually requires anatomical understanding, proportion measurement, and careful estimation of head orientation. Artists often struggle to achieve consistent accuracy, especially when drawing from photographic references.

This project aims to remove that difficulty by using AI to automatically detect the face, compute head geometry, determine orientation, and draw Loomis Method guidelines directly onto the reference image. This provides accurate, consistent construction lines that help artists learn and apply the Loomis method correctly.

2. Solution Overview

The Loomis Drawing Assistant is an AI-driven system that:

Detects facial landmarks using MediaPipe Face Mesh (468 points)

Calculates head proportions, center, radius, and face direction

Draws Loomis construction guidelines including:

Loomis sphere

Side-plane ellipse

Vertical and horizontal axes

Brow line

Nose and chin indicators

Jawline curve

Outer face contour

Provides step-by-step instructions (via LLM agent)

Supports tool-based execution via LangChain agent and MCP server

Uses a local LLM through Ollama (no API key required)

This automates the foundational steps of the Loomis drawing method, enabling artists to focus on actual sketching and learning rather than manual measurement.

3. System Architecture
                     User Input (text prompt)
                                |
                                v
                         agent.py (LLM)
                    LangChain ReAct Agent
                                |
        --------------------------------------------------
        |                        |                       |
        v                        v                       v
 detect_face tool      apply_loomis tool     analyze_proportions tool
        |                        |                       |
        v                        v                       v
 MediaPipe CV           render_steps.py            geometry_utils.py
 FaceMesh, Pose         Drawing guidelines         Mathematical analysis
        |
        v
  Image annotated with Loomis lines

Major Components
File	Purpose
agent.py	LLM agent using tools and reasoning (ReAct)
mcp_server.py	MCP tool server for integration with IDEs (VSCode, Cursor, etc.)
pose_detection.py	MediaPipe-based landmark extraction
geometry_utils.py	Head geometry calculations, proportions, angles
render_steps.py	Drawing Loomis construction lines on the image
requirements.txt	Dependencies
4. Key Features Demonstrated (Required for Kaggle Capstone)

This project includes several advanced agent features required by the competition:

Multi-Agent / Tool-Based Framework

ReAct-style agent using LangGraph

Multiple custom tools exposed to the LLM

Custom Tools

detect_face

apply_loomis

analyze_proportions

MCP Integration

Full MCP server implementation enabling tool-based execution directly through supporting environments

Local LLM Behind the Agent

Uses Ollama with a local qwen2.5 model

Context Engineering

Geometry extracted from media, passed through the agent for reasoning

These features satisfy more than three required concepts for full implementation points.

5. Installation and Setup
Step 1: Clone the Repository
git clone https://github.com/yourname/Loomis_Drawing_Assistant.git
cd Loomis_Drawing_Assistant

Step 2: Create a Virtual Environment
python -m venv venv
venv\Scripts\activate      (Windows)
source venv/bin/activate   (Mac/Linux)

Step 3: Install Python Requirements
pip install -r requirements.txt

Step 4: Install Ollama (if not installed)

Download from:
https://ollama.com/

Step 5: Pull the LLM Model
ollama pull qwen2.5:1.5b

Step 6: Run the Agent
python agent.py

Step 7: (Optional) Run MCP Server
python mcp_server.py

6. Usage Examples
Detect Face
Detect face in D:/images/photo.jpg

Apply Loomis Construction Lines
Apply Loomis to D:/images/photo.jpg and save to D:/output/loomis.jpg

Analyze Facial Proportions
Analyze proportions in D:/images/photo.jpg

7. Output Examples

Include here before/after images:

Original reference photo

Output image with:

Loomis sphere

Side-plane ellipse

Brow line

Center axis

Jaw curve

Chin and nose markers

8. Value Proposition

Reduces time spent constructing foundation lines manually

Improves accuracy of head drawings using real measured proportions

Helps artists learn and internalize the Loomis method more quickly

Useful for both education and production sketching workflows

Fully automated and consistent across images

This assistant provides a significant improvement over manual Loomis construction by leveraging modern CV and geometry analysis combined with LLM reasoning.
