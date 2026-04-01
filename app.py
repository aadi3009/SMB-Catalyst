import os
import json
import math
from datetime import datetime
from pathlib import Path

from phase1 import extract_architectural_program
from phase2 import Phase2LayoutEngine
from phase3 import Phase3DraftingEngine
from phase4 import Phase4DXFExporter


# ============================================================
# 1. CONFIG
# ============================================================

OUTPUT_DIR = Path("outputs")
DEFAULT_BRIEF = "3BHK, 1500 sq ft, open kitchen, 1 study, maximize natural light"
DEFAULT_NUM_CANDIDATES = 220
DEFAULT_SEED = 42


# ============================================================
# 2. HELPERS
# ============================================================

def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def timestamp_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def choose_plot_width(total_area: float) -> int:
    """
    Keeps plot roughly square-ish for better layouts.
    """
    return max(24, int(round(math.sqrt(total_area))))


def convert_phase1_to_phase2_requirements(project_brief) -> list[dict]:
    """
    Converts Phase 1 Pydantic object into the list-of-dicts
    format expected by Phase 2.
    """
    requirements = []
    for room in project_brief.rooms:
        requirements.append({
            "name": room.name,
            "room_type": room.room_type,
            "min_area_sqft": float(room.min_area_sqft),
            "adjacencies": list(room.adjacencies),
        })
    return requirements


def summarize_program(project_brief) -> dict:
    return {
        "total_area": float(project_brief.total_area),
        "bhk_config": int(project_brief.bhk_config),
        "special_constraints": list(project_brief.special_constraints),
        "rooms": [
            {
                "name": room.name,
                "room_type": room.room_type,
                "min_area_sqft": float(room.min_area_sqft),
                "adjacencies": list(room.adjacencies),
            }
            for room in project_brief.rooms
        ],
    }


def save_json(data: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def print_program_summary(program_dict: dict) -> None:
    room_sum = sum(r["min_area_sqft"] for r in program_dict["rooms"])
    total_area = program_dict["total_area"]
    ratio = room_sum / total_area if total_area else 0.0

    print("\n--- Extracted Program Summary ---")
    print(f"Total area         : {total_area}")
    print(f"BHK config         : {program_dict['bhk_config']}")
    print(f"Room count         : {len(program_dict['rooms'])}")
    print(f"Sum of room areas  : {room_sum:.1f}")
    print(f"Area ratio         : {ratio:.2f}")
    print(f"Constraints        : {program_dict['special_constraints']}")


# ============================================================
# 3. MAIN PIPELINE
# ============================================================

def generate_dxf_from_brief(
    brief: str,
    output_basename: str | None = None,
    show_preview: bool = True,
    num_candidates: int = DEFAULT_NUM_CANDIDATES,
    seed: int = DEFAULT_SEED,
) -> dict:
    """
    End-to-end pipeline:
    brief -> phase1 -> phase2 -> phase3 -> phase4 -> DXF

    Returns a dict with useful output paths and metadata.
    """
    ensure_output_dir()

    run_id = output_basename or f"layout_{timestamp_str()}"
    json_path = OUTPUT_DIR / f"{run_id}_program.json"
    dxf_path = OUTPUT_DIR / f"{run_id}.dxf"

    # -------------------------
    # Phase 1: Extract program
    # -------------------------
    print("\n[Phase 1] Extracting structured architectural program...")
    extracted_program = extract_architectural_program(brief)
    program_dict = summarize_program(extracted_program)
    save_json(program_dict, json_path)
    print_program_summary(program_dict)

    total_area = float(extracted_program.total_area)
    requirements = convert_phase1_to_phase2_requirements(extracted_program)

    # -------------------------
    # Plot selection
    # -------------------------
    plot_width = choose_plot_width(total_area)
    plot_height = total_area / plot_width
    print("\n--- Plot Assumption ---")
    print(f"Plot width  : {plot_width} ft")
    print(f"Plot height : {plot_height:.2f} ft")

    # -------------------------
    # Phase 2: Layout
    # -------------------------
    print("\n[Phase 2] Generating layout...")
    phase2 = Phase2LayoutEngine(
        total_area=total_area,
        plot_width=plot_width,
        seed=seed,
    )

    best_layout, best_score, breakdown = phase2.generate_best_layout(
        requirements,
        num_candidates=num_candidates,
        verbose=True,
    )

    print(f"\nBest layout score: {best_score:.2f}")

    # -------------------------
    # Phase 3: Drafting scene
    # -------------------------
    print("\n[Phase 3] Building drafting scene...")
    drafting = Phase3DraftingEngine(
        plot_width=plot_width,
        plot_height=plot_height,
        exterior_wall_thickness=0.75,
        interior_wall_thickness=0.375,
    )

    scene = drafting.build_drafting_scene(best_layout, requirements)
    drafting.print_accessibility_report(scene)

    if show_preview:
        drafting.visualize(scene, title="End-to-End Generated Plan Preview")

    # -------------------------
    # Phase 4: DXF export
    # -------------------------
    print("\n[Phase 4] Exporting DXF...")
    exporter = Phase4DXFExporter(units="feet")
    saved_path = exporter.export_scene_to_dxf(scene, str(dxf_path))
    #exporter.print_export_summary(scene, saved_path)

    result = {
        "brief": brief,
        "program_json": str(json_path.resolve()),
        "dxf_file": str(Path(saved_path).resolve()),
        "total_area": total_area,
        "plot_width": plot_width,
        "plot_height": plot_height,
        "best_score": best_score,
        "num_candidates": num_candidates,
    }

    return result


# ============================================================
# 4. CLI ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    print("=== Automated Layout & Drafting System ===")
    print("Paste a client brief. Press Enter on an empty line to use the default example.\n")

    user_brief = input("Client brief: ").strip()
    if not user_brief:
        user_brief = DEFAULT_BRIEF
        print(f"\nUsing default brief:\n{user_brief}")

    try:
        result = generate_dxf_from_brief(
            brief=user_brief,
            show_preview=True,
            num_candidates=DEFAULT_NUM_CANDIDATES,
            seed=DEFAULT_SEED,
        )

        print("\n=== Pipeline Complete ===")
        print(f"Program JSON : {result['program_json']}")
        print(f"DXF file     : {result['dxf_file']}")
        print(f"Best score   : {result['best_score']:.2f}")

    except Exception as e:
        print("\n=== Pipeline Failed ===")
        print(f"Error: {e}")
        print("\nChecks:")
        print("1. OPENAI_API_KEY is set correctly")
        print("2. phase1.py, phase2.py, phase3.py, phase4.py are in the same folder")
        print("3. Required packages are installed: openai, pydantic, matplotlib, ezdxf")