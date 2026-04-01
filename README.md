# Automated Layout & Drafting System (Case B)

This project is a functional prototype developed for the **SMB Catalyst x IIT Bombay Campus Recruitment 2026**. It automates the transition from unstructured architectural briefs to editable, CAD-ready DXF floor plans.

## Project Overview
The system processes natural language inputs (e.g., "3BHK, 1500 sq ft, open kitchen") and generates a zoned floor plan with room dimensions, spatial allocation, door/window placement, and basic furniture layouts.

### End-to-End Pipeline
1.  **Phase 1: Input Processing** (`phase1.py`)
    * Uses the **Groq API** (Llama-based models) to extract structured requirements from unstructured text.
    * Defines a strict Pydantic schema for rooms, areas, and adjacencies.
2.  **Phase 2: Layout Engine** (`phase2.py`)
    * Implements a heuristic-based spatial optimization engine.
    * Generates and scores multiple candidates based on zoning (public/private/service), aspect ratios, and adjacency rewards.
3.  **Phase 4: Drafting Engine** (`phase3.py`)
    * Converts abstract room rectangles into technical geometry.
    * Handles wall thicknesses, accessibility-aware door placement, and window/ventilator logic.
    * Auto-populates rooms with standard furniture blocks.
4.  **Phase 4: Output Generation** (`phase4.py`)
    * Serializes the drafting scene into a standardized **DXF file** using the `ezdxf` library.
    * Organizes geometry into professional CAD layers (WALL_EXT, WALL_INT, DOORS, WINDOWS, FURNITURE, etc.).

## Prerequisites
* Python 3.10+
* Groq API Key (Set as `GROQ_API_KEY` environment variable)

### Required Libraries
```bash
pip install groq pydantic matplotlib ezdxf
```

## How to Run
Run the main application script to start the interactive pipeline:
```bash
python app.py
```
You will be prompted to enter a client brief. After processing, the script will generate:
* A structured JSON program in the `outputs/` folder.
* A visual preview of the layout using Matplotlib.
* An editable `.dxf` file in the `outputs/` folder.

## Key Features & Trade-offs
* **Hybrid Approach:** Combines LLM-based intent extraction with deterministic geometry rules for reliability and editability.
* **Accessibility Logic:** Includes a connectivity check to ensure all rooms are reachable from the main entry.
* **Human Override:** The DXF output is layer-separated, allowing architects to manually refine the "first-pass" design in AutoCAD or Revit.

---
**Author:** Aaditya Nagare
**Case Study:** Case B - Automated Layout & Drafting System
