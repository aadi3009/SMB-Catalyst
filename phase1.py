import os
import json
from typing import List
from pydantic import BaseModel, Field, ConfigDict
from groq import Groq

# ============================================================
# 1. CLIENT SETUP
# ============================================================
# Ensure your GROQ_API_KEY is set in your environment
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Recommended model for high-speed structured extraction
MODEL_NAME = "openai/gpt-oss-20b"

# ============================================================
# 2. PYDANTIC SCHEMA (GROQ STRICT MODE)
# ============================================================

class RoomRequirement(BaseModel):
    # 'extra="forbid"' ensures the JSON schema includes 'additionalProperties: false'
    model_config = ConfigDict(extra='forbid')
    
    name: str = Field(description="Name of the room (e.g., Living Room, Kitchen)")
    room_type: str = Field(description="Categorize as: public, private, or service")
    min_area_sqft: float = Field(description="Minimum area required in square feet")
    adjacencies: List[str] = Field(description="Names of other rooms this room must connect to")


class ProjectBrief(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    total_area: float = Field(description="Total plot or apartment area in square feet")
    bhk_config: int = Field(description="Number of primary bedrooms")
    rooms: List[RoomRequirement] = Field(
        description="Complete list of all rooms including implied necessary rooms like bathrooms and kitchens"
    )
    special_constraints: List[str] = Field(
        description="Any specific architectural or design constraints mentioned"
    )

# ============================================================
# 3. EXTRACTION ENGINE
# ============================================================

def extract_architectural_program(client_brief: str) -> ProjectBrief:
    """
    Translates an unstructured client brief into a structured Requirement Graph.
    Uses standard Indian architectural norms for missing area values.
    """
    system_prompt = (
        "You are a Senior Architectural Analyst for SMB Catalyst. "
        "Extract structured requirements from the provided brief. "
        "If room areas are missing, use standard Indian norms (e.g., Kitchen ~90sqft) & use the unit as sqft. "
        "If a BHK is mentioned, ensure you include necessary bathrooms and a living area. "
        "Output strictly follows the provided JSON schema."
    )

    # Generate the JSON Schema from Pydantic for Groq's validator
    json_schema = ProjectBrief.model_json_schema()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": client_brief},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "ArchitecturalProgram",
                "schema": json_schema,
                "strict": True,
            },
        },
        temperature=0.1,  # Low temperature for deterministic data extraction
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("API returned empty content")

    # Final validation: Ensures the LLM output perfectly matches the Python model
    return ProjectBrief.model_validate_json(content)

# ============================================================
# 4. EXECUTION DEMO
# ============================================================

if __name__ == "__main__":
    # Standard test case from the Case Study [cite: 70]
    sample_brief = (
        "3BHK, 1500 sq ft, open kitchen, 1 study, maximize natural light. "
        "I want the kitchen right next to the dining area."
    )

    print("--- Phase 1: Processing Unstructured Brief ---")
    try:
        program = extract_architectural_program(sample_brief)
        print(program.model_dump_json(indent=2))
        
        # This 'program' object is now ready for Phase 2: Layout Engine
    except Exception as e:
        print(f"Extraction Error: {e}")