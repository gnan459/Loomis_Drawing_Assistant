# Loomis Drawing Assistant

AI-powered geometric construction system for the Loomis head-drawing
method with visual memory using Qdrant

## 1. Overview

Constructing a realistic human head requires accurate proportions. The
Loomis Method breaks the head into simple geometric forms, but applying
these guidelines manually from a reference image is difficult and slow.\
This project automates the Loomis construction process by detecting
facial landmarks, computing precise geometric measurements, storing and
retrieving geometry-based visual memory using Qdrant, and generating
guiding lines and step-by-step drawing instructions directly on the
image.

## 2. Problem Statement

Beginners struggle with: - Understanding correct head proportions -
Determining head orientation and tilt - Manually drawing the Loomis
sphere and side planes - Accurately placing brow, nose, chin, and jaw
guidelines - Maintaining consistency across different references -
Learning from past mistakes without structured feedback

## 3. Solution Summary

The Loomis Drawing Assistant: - Detects facial landmarks using MediaPipe
Face Mesh - Computes head geometry (radius, center, orientation) -
Automatically draws Loomis construction guides - Uses a ReAct-based LLM
agent for tool orchestration - Uses a second Instructor LLM agent to
generate step-by-step drawing tutorials - Stores compact geometry
embeddings in Qdrant as visual memory - Retrieves similar past
constructions to condition drawing guidance - Runs entirely locally
using Ollama + Qdrant

## 4. Architecture

User Input → LLM Agent → MediaPipe → Geometry Utils → Qdrant →
Instructor LLM → Output

## 5. Qdrant Integration

Qdrant is used as a persistent visual memory system. Geometry embeddings
derived from face landmarks are stored and retrieved to provide
pose-conditioned Loomis construction guidance and comparative feedback.

## 6. Installation & Setup

``` bash
git clone https://github.com/gnan459/Loomis_Drawing_Assistant.git
cd Loomis_Drawing_Assistant
pip install -r requirements.txt
docker run -p 6333:6333 qdrant/qdrant
ollama pull qwen2.5:1.5b
python agent.py
```

## 7. Usage Conclusion

A complete AI pipeline integrating computer vision, geometry, vector
memory, and LLM reasoning for art education.
