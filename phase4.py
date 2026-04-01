import os
from typing import Dict, List, Tuple

import ezdxf

from phase2 import Phase2LayoutEngine
from phase3 import Phase3DraftingEngine


# ============================================================
# 1. DXF EXPORTER
# ============================================================

class Phase4DXFExporter:
    """
    Converts the Phase 3 drafting scene into a DXF file.

    Output layers:
    - WALL_EXT
    - WALL_INT
    - DOORS
    - WINDOWS
    - FURNITURE
    - ROOM_TEXT
    - PLOT_BOUNDARY
    """

    def __init__(self, units: str = "feet"):
        self.units = units.lower()

    # --------------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------------

    def export_scene_to_dxf(self, scene: Dict, output_path: str) -> str:
        doc = ezdxf.new("R2010")
        self._set_units(doc)

        self._create_layers(doc)

        msp = doc.modelspace()

        # Plot boundary
        self._add_plot_boundary(msp, scene)

        # Walls
        self._add_walls(msp, scene)

        # Doors
        self._add_doors(msp, scene)

        # Windows
        self._add_windows(msp, scene)

        # Furniture
        self._add_furniture(msp, scene)

        # Room labels
        self._add_room_labels(msp, scene)

        # Save
        doc.saveas(output_path)
        return output_path

    # --------------------------------------------------------
    # DXF SETUP
    # --------------------------------------------------------

    def _set_units(self, doc):
        """
        DXF INSUNITS codes:
        0 = Unitless
        1 = Inches
        2 = Feet
        4 = Millimeters
        6 = Meters
        """
        unit_map = {
            "unitless": 0,
            "inches": 1,
            "feet": 2,
            "mm": 4,
            "millimeters": 4,
            "meters": 6,
        }
        doc.header["$INSUNITS"] = unit_map.get(self.units, 2)

    def _create_layers(self, doc):
        layers = [
            ("WALL_EXT", 7),       # white / black depending on theme
            ("WALL_INT", 8),       # gray
            ("DOORS", 30),         # orange-ish
            ("WINDOWS", 5),        # blue
            ("FURNITURE", 9),      # light gray
            ("ROOM_TEXT", 2),      # yellow
            ("PLOT_BOUNDARY", 3),  # green
        ]

        existing = {layer.dxf.name for layer in doc.layers}
        for name, color in layers:
            if name not in existing:
                doc.layers.add(name=name, color=color)

    # --------------------------------------------------------
    # GEOMETRY HELPERS
    # --------------------------------------------------------

    def _rect_points(self, x: float, y: float, w: float, h: float) -> List[Tuple[float, float]]:
        return [
            (x, y),
            (x + w, y),
            (x + w, y + h),
            (x, y + h),
            (x, y),
        ]

    def _add_rect_polyline(self, msp, x: float, y: float, w: float, h: float, layer: str):
        points = self._rect_points(x, y, w, h)
        msp.add_lwpolyline(points, dxfattribs={"layer": layer, "closed": True})

    # --------------------------------------------------------
    # PLOT BOUNDARY
    # --------------------------------------------------------

    def _add_plot_boundary(self, msp, scene: Dict):
        layout = scene["layout"]
        if not layout:
            return

        # infer plot extents from layout scene
        max_x = max(r.x + r.w for r in layout)
        max_y = max(r.y + r.h for r in layout)

        self._add_rect_polyline(msp, 0.0, 0.0, max_x, max_y, "PLOT_BOUNDARY")

    # --------------------------------------------------------
    # WALLS
    # --------------------------------------------------------

    def _add_walls(self, msp, scene: Dict):
        walls = scene["walls"]

        for wall in walls:
            layer = "WALL_EXT" if wall.wall_type == "exterior" else "WALL_INT"
            self._add_rect_polyline(msp, wall.x, wall.y, wall.w, wall.h, layer)

    # --------------------------------------------------------
    # DOORS
    # --------------------------------------------------------

    def _add_doors(self, msp, scene: Dict):
        doors = scene["doors"]

        for door in doors:
            # opening marker rectangle
            self._add_rect_polyline(msp, door.x, door.y, door.w, door.h, "DOORS")

            # center marker line to make it readable in CAD
            if door.orientation == "vertical":
                cx = door.x + door.w / 2
                msp.add_line(
                    (cx, door.y),
                    (cx, door.y + door.h),
                    dxfattribs={"layer": "DOORS"},
                )
            else:
                cy = door.y + door.h / 2
                msp.add_line(
                    (door.x, cy),
                    (door.x + door.w, cy),
                    dxfattribs={"layer": "DOORS"},
                )

    # --------------------------------------------------------
    # WINDOWS
    # --------------------------------------------------------

    def _add_windows(self, msp, scene: Dict):
        windows = scene["windows"]

        for win in windows:
            self._add_rect_polyline(msp, win.x, win.y, win.w, win.h, "WINDOWS")

            if win.orientation == "vertical":
                cx = win.x + win.w / 2
                msp.add_line(
                    (cx, win.y),
                    (cx, win.y + win.h),
                    dxfattribs={"layer": "WINDOWS"},
                )
            else:
                cy = win.y + win.h / 2
                msp.add_line(
                    (win.x, cy),
                    (win.x + win.w, cy),
                    dxfattribs={"layer": "WINDOWS"},
                )

    # --------------------------------------------------------
    # FURNITURE
    # --------------------------------------------------------

    def _add_furniture(self, msp, scene: Dict):
        furniture = scene["furniture"]

        for item in furniture:
            self._add_rect_polyline(msp, item.x, item.y, item.w, item.h, "FURNITURE")

            text_height = max(0.5, min(item.w, item.h) * 0.18)
            msp.add_text(
                item.item_type,
                dxfattribs={
                    "layer": "FURNITURE",
                    "height": text_height,
                    "insert": (item.x + item.w / 2, item.y + item.h / 2),
                    "halign": 1,   # center
                    "valign": 2,   # middle
                },
            )

    # --------------------------------------------------------
    # ROOM LABELS
    # --------------------------------------------------------

    def _label_anchor(self, drafting_engine, room) -> Tuple[float, float]:
        return drafting_engine._label_anchor(room)

    def _add_room_labels(self, msp, scene: Dict):
        layout = scene["layout"]

        # use same label logic as phase3 for consistency
        # lightweight trick: instantiate tiny helper from scene inference
        max_x = max(r.x + r.w for r in layout)
        max_y = max(r.y + r.h for r in layout)
        drafting_engine = Phase3DraftingEngine(max_x, max_y)

        for room in layout:
            lx, ly = self._label_anchor(drafting_engine, room)

            text = (
                f"{room.name}\\P"
                f"{int(round(room.area))} sqft\\P"
                f"{room.w:.1f}' x {room.h:.1f}'"
            )

            # MTEXT supports line breaks nicely
            msp.add_mtext(
                text,
                dxfattribs={
                    "layer": "ROOM_TEXT",
                    "char_height": 0.8,
                    "insert": (lx, ly),
                },
            )


# ============================================================
# 2. EXAMPLE END-TO-END USAGE
# ============================================================

if __name__ == "__main__":
    requirements = [
        {"name": "Living Room", "room_type": "public", "min_area_sqft": 300, "adjacencies": ["Dining Area", "Study Room"]},
        {"name": "Dining Area", "room_type": "public", "min_area_sqft": 150, "adjacencies": ["Living Room", "Kitchen"]},
        {"name": "Master Bedroom", "room_type": "private", "min_area_sqft": 200, "adjacencies": ["Bathroom 1", "Bedroom 2"]},
        {"name": "Bedroom 2", "room_type": "private", "min_area_sqft": 150, "adjacencies": ["Bathroom 1"]},
        {"name": "Bedroom 3", "room_type": "private", "min_area_sqft": 140, "adjacencies": ["Bathroom 2"]},
        {"name": "Study Room", "room_type": "private", "min_area_sqft": 100, "adjacencies": ["Living Room"]},
        {"name": "Kitchen", "room_type": "service", "min_area_sqft": 120, "adjacencies": ["Dining Area"]},
        {"name": "Bathroom 1", "room_type": "service", "min_area_sqft": 55, "adjacencies": ["Master Bedroom", "Bedroom 2"]},
        {"name": "Bathroom 2", "room_type": "service", "min_area_sqft": 50, "adjacencies": ["Bedroom 3"]},
    ]

    total_area = 1500
    plot_width = 40
    output_file = "floorplan_output.dxf"

    # -------------------------
    # Phase 2
    # -------------------------
    phase2 = Phase2LayoutEngine(
        total_area=total_area,
        plot_width=plot_width,
        seed=42
    )

    best_layout, best_score, breakdown = phase2.generate_best_layout(
        requirements,
        num_candidates=220,
        verbose=True
    )

    # -------------------------
    # Phase 3
    # -------------------------
    plot_height = total_area / plot_width

    drafting = Phase3DraftingEngine(
        plot_width=plot_width,
        plot_height=plot_height,
        exterior_wall_thickness=0.75,
        interior_wall_thickness=0.375,
    )

    scene = drafting.build_drafting_scene(best_layout, requirements)
    drafting.print_accessibility_report(scene)

    # optional preview
    drafting.visualize(scene, title="Phase 3 Preview Before DXF Export")

    # -------------------------
    # Phase 4
    # -------------------------
    exporter = Phase4DXFExporter(units="feet")
    saved_path = exporter.export_scene_to_dxf(scene, output_file)

    print(f"\nDXF exported successfully: {os.path.abspath(saved_path)}")