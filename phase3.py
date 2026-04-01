import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, deque

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from phase2 import Phase2LayoutEngine


# ============================================================
# 1. DATA CLASSES
# ============================================================

@dataclass
class WallRect:
    x: float
    y: float
    w: float
    h: float
    wall_type: str
    orientation: str
    room_a: Optional[str] = None
    room_b: Optional[str] = None


@dataclass
class Opening:
    x: float
    y: float
    w: float
    h: float
    opening_type: str       # door / window
    orientation: str        # vertical / horizontal
    host_wall_type: str     # interior / exterior
    room_a: Optional[str] = None
    room_b: Optional[str] = None
    style: str = "standard" # standard / open_passage / main_door / ventilator


@dataclass
class FurnitureItem:
    room_name: str
    item_type: str
    x: float
    y: float
    w: float
    h: float
    rotation: float = 0.0


# ============================================================
# 2. PHASE 3 DRAFTING ENGINE
# ============================================================

class Phase3DraftingEngine:
    def __init__(
        self,
        plot_width: float,
        plot_height: float,
        exterior_wall_thickness: float = 0.75,
        interior_wall_thickness: float = 0.375,
    ):
        self.plot_width = plot_width
        self.plot_height = plot_height
        self.ext_t = exterior_wall_thickness
        self.int_t = interior_wall_thickness

        self.default_adjacency_rules = {
            "Living Room": ["Dining Area", "Study", "Study Room"],
            "Dining Area": ["Living Room", "Kitchen"],
            "Kitchen": ["Dining Area"],
            "Master Bedroom": ["Bathroom 1", "Bedroom 2"],
            "Bedroom 1": ["Bathroom 1", "Bedroom 2"],
            "Bedroom 2": ["Bathroom 1", "Bedroom 3", "Master Bedroom", "Bedroom 1"],
            "Bedroom 3": ["Bathroom 2"],
            "Study Room": ["Living Room"],
            "Study": ["Living Room"],
            "Bathroom 1": ["Master Bedroom", "Bedroom 1", "Bedroom 2"],
            "Bathroom 2": ["Bedroom 3"],
        }

    # --------------------------------------------------------
    # ROOM-TYPE NAME HELPERS
    # --------------------------------------------------------

    def _is_living(self, name: str) -> bool:
        return "living" in name.lower()

    def _is_dining(self, name: str) -> bool:
        return "dining" in name.lower()

    def _is_kitchen(self, name: str) -> bool:
        return "kitchen" in name.lower()

    def _is_study(self, name: str) -> bool:
        return "study" in name.lower()

    def _is_bedroom(self, name: str) -> bool:
        n = name.lower()
        return "bedroom" in n or "master" in n or n.startswith("bed ")

    def _is_bathroom(self, name: str) -> bool:
        n = name.lower()
        return "bath" in n or "toilet" in n or "wc" in n

    # --------------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------------

    def build_drafting_scene(self, layout, requirements: List[Dict]) -> Dict:
        room_map = {r.name: r for r in layout}
        adjacency_weights = self._build_adjacency_weights(requirements)
        shared_map = self._build_shared_wall_map(layout)

        walls = self._generate_walls(layout, shared_map)

        entry_room = self._pick_entry_room(layout)
        main_door = self._place_main_door(room_map[entry_room])

        desired_doors = self._place_adjacency_doors(layout, shared_map, adjacency_weights)
        all_doors = [main_door] + desired_doors
        all_doors = self._ensure_accessibility(layout, shared_map, all_doors, entry_room)

        windows = self._place_windows(layout, all_doors)
        furniture = self._place_furniture(layout, all_doors)

        return {
            "layout": layout,
            "requirements": requirements,
            "room_map": room_map,
            "walls": walls,
            "doors": all_doors,
            "windows": windows,
            "furniture": furniture,
            "shared_map": shared_map,
            "entry_room": entry_room,
        }

    def visualize(self, scene: Dict, title: str = "Phase 3 Drafting Engine Output") -> None:
        layout = scene["layout"]
        walls = scene["walls"]
        doors = scene["doors"]
        windows = scene["windows"]
        furniture = scene["furniture"]

        fig, ax = plt.subplots(figsize=(15, 10))
        margin = max(self.ext_t * 2, 1.0)

        ax.set_xlim(-margin - self.ext_t, self.plot_width + margin + self.ext_t)
        ax.set_ylim(-margin - self.ext_t, self.plot_height + margin + self.ext_t)
        ax.set_aspect("equal")

        room_colors = {
            "public": "#d9edf7",
            "private": "#dff0d8",
            "service": "#fcf8e3",
            "hybrid_bath": "#f5efe1",
        }

        for room in layout:
            fill = patches.Rectangle(
                (room.x, room.y),
                room.w,
                room.h,
                facecolor=room_colors.get(room.room_type, "whitesmoke"),
                edgecolor="none",
                alpha=0.45,
                zorder=1,
            )
            ax.add_patch(fill)

        for wall in walls:
            color = "#2f2f2f" if wall.wall_type == "exterior" else "#555555"
            patch = patches.Rectangle(
                (wall.x, wall.y),
                wall.w,
                wall.h,
                facecolor=color,
                edgecolor=color,
                linewidth=1.0,
                zorder=5,
            )
            ax.add_patch(patch)

        for op in doors + windows:
            carve = patches.Rectangle(
                (op.x, op.y),
                op.w,
                op.h,
                facecolor="white",
                edgecolor="white",
                linewidth=0,
                zorder=6,
            )
            ax.add_patch(carve)

        for door in doors:
            color = "#8b4513" if door.style not in {"open_passage", "main_door"} else "#c97f1f"
            lw = 2.4 if door.style != "main_door" else 3.2
            if door.orientation == "vertical":
                ax.plot(
                    [door.x + door.w / 2, door.x + door.w / 2],
                    [door.y, door.y + door.h],
                    color=color, linewidth=lw, zorder=7
                )
            else:
                ax.plot(
                    [door.x, door.x + door.w],
                    [door.y + door.h / 2, door.y + door.h / 2],
                    color=color, linewidth=lw, zorder=7
                )

        for win in windows:
            color = "#1f77b4"
            if win.orientation == "vertical":
                ax.plot(
                    [win.x + win.w / 2, win.x + win.w / 2],
                    [win.y, win.y + win.h],
                    color=color, linewidth=3.0, zorder=7
                )
            else:
                ax.plot(
                    [win.x, win.x + win.w],
                    [win.y + win.h / 2, win.y + win.h / 2],
                    color=color, linewidth=3.0, zorder=7
                )

        for f in furniture:
            patch = patches.Rectangle(
                (f.x, f.y),
                f.w,
                f.h,
                facecolor="#cfcfcf",
                edgecolor="#666666",
                linewidth=1.0,
                alpha=0.9,
                zorder=3,
            )
            ax.add_patch(patch)

            ax.text(
                f.x + f.w / 2,
                f.y + f.h / 2,
                f.item_type,
                ha="center",
                va="center",
                fontsize=7,
                zorder=4,
            )

        for room in layout:
            lx, ly = self._label_anchor(room)
            label = (
                f"{room.name}\n"
                f"{int(round(room.area))} sqft\n"
                f"{room.w:.1f}' x {room.h:.1f}'"
            )
            ax.text(
                lx,
                ly,
                label,
                ha="center",
                va="center",
                fontsize=10,
                weight="bold",
                zorder=8,
            )

        boundary = patches.Rectangle(
            (0, 0),
            self.plot_width,
            self.plot_height,
            fill=False,
            edgecolor="black",
            linestyle="--",
            linewidth=1.2,
            zorder=2,
        )
        ax.add_patch(boundary)

        ax.grid(True, linestyle="--", alpha=0.25)
        ax.set_xlabel("Feet", fontsize=14)
        ax.set_ylabel("Feet", fontsize=14)
        ax.set_title(title, fontsize=22)
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------
    # BASIC HELPERS
    # --------------------------------------------------------

    def _build_adjacency_weights(self, requirements: List[Dict]) -> Dict[Tuple[str, str], float]:
        names = {r["name"] for r in requirements}
        weights: Dict[Tuple[str, str], float] = {}

        def add_weight(a: str, b: str, w: float):
            if a == b:
                return
            key = tuple(sorted((a, b)))
            weights[key] = max(weights.get(key, 0.0), w)

        for r in requirements:
            for other in r.get("adjacencies", []):
                if other in names:
                    add_weight(r["name"], other, 14.0)

        for r in requirements:
            for other in self.default_adjacency_rules.get(r["name"], []):
                if other in names:
                    add_weight(r["name"], other, 5.0)

        return weights

    def _interval_overlap(self, a1: float, a2: float, b1: float, b2: float) -> float:
        return max(0.0, min(a2, b2) - max(a1, b1))

    def _shared_edge_info(self, a, b):
        eps = 1e-6

        if abs((a.x + a.w) - b.x) < eps or abs((b.x + b.w) - a.x) < eps:
            y1 = max(a.y, b.y)
            y2 = min(a.y + a.h, b.y + b.h)
            overlap = y2 - y1
            if overlap > 1e-6:
                x_common = a.x + a.w if abs((a.x + a.w) - b.x) < eps else b.x + b.w
                return {
                    "orientation": "vertical",
                    "shared_length": overlap,
                    "x": x_common,
                    "y": y1,
                }

        if abs((a.y + a.h) - b.y) < eps or abs((b.y + b.h) - a.y) < eps:
            x1 = max(a.x, b.x)
            x2 = min(a.x + a.w, b.x + b.w)
            overlap = x2 - x1
            if overlap > 1e-6:
                y_common = a.y + a.h if abs((a.y + a.h) - b.y) < eps else b.y + b.h
                return {
                    "orientation": "horizontal",
                    "shared_length": overlap,
                    "x": x1,
                    "y": y_common,
                }

        return None

    def _build_shared_wall_map(self, layout) -> Dict[Tuple[str, str], Dict]:
        shared = {}
        for i in range(len(layout)):
            for j in range(i + 1, len(layout)):
                a = layout[i]
                b = layout[j]
                info = self._shared_edge_info(a, b)
                if info:
                    shared[(a.name, b.name)] = info
        return shared

    def _touches_boundary_left(self, room) -> bool:
        return abs(room.x - 0.0) < 1e-6

    def _touches_boundary_right(self, room) -> bool:
        return abs((room.x + room.w) - self.plot_width) < 1e-6

    def _touches_boundary_bottom(self, room) -> bool:
        return abs(room.y - 0.0) < 1e-6

    def _touches_boundary_top(self, room) -> bool:
        return abs((room.y + room.h) - self.plot_height) < 1e-6

    def _rects_overlap(self, a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)

    # --------------------------------------------------------
    # OPENING COLLISION HELPERS
    # --------------------------------------------------------

    def _openings_conflict(self, op1: Opening, op2: Opening, clearance: float = 1.5) -> bool:
        if op1.orientation != op2.orientation:
            return False

        if op1.orientation == "horizontal":
            if abs(op1.y - op2.y) > 1e-6:
                return False
            a1, a2 = op1.x - clearance, op1.x + op1.w + clearance
            b1, b2 = op2.x - clearance, op2.x + op2.w + clearance
            return not (a2 <= b1 or b2 <= a1)
        else:
            if abs(op1.x - op2.x) > 1e-6:
                return False
            a1, a2 = op1.y - clearance, op1.y + op1.h + clearance
            b1, b2 = op2.y - clearance, op2.y + op2.h + clearance
            return not (a2 <= b1 or b2 <= a1)

    # --------------------------------------------------------
    # WALLS
    # --------------------------------------------------------

    def _generate_walls(self, layout, shared_map) -> List[WallRect]:
        walls: List[WallRect] = []

        walls.append(WallRect(0.0, -self.ext_t, self.plot_width, self.ext_t, "exterior", "horizontal"))
        walls.append(WallRect(0.0, self.plot_height, self.plot_width, self.ext_t, "exterior", "horizontal"))
        walls.append(WallRect(-self.ext_t, 0.0, self.ext_t, self.plot_height, "exterior", "vertical"))
        walls.append(WallRect(self.plot_width, 0.0, self.ext_t, self.plot_height, "exterior", "vertical"))

        for (room_a, room_b), info in shared_map.items():
            if info["orientation"] == "vertical":
                walls.append(WallRect(
                    info["x"] - self.int_t / 2,
                    info["y"],
                    self.int_t,
                    info["shared_length"],
                    "interior",
                    "vertical",
                    room_a,
                    room_b,
                ))
            else:
                walls.append(WallRect(
                    info["x"],
                    info["y"] - self.int_t / 2,
                    info["shared_length"],
                    self.int_t,
                    "interior",
                    "horizontal",
                    room_a,
                    room_b,
                ))
        return walls

    # --------------------------------------------------------
    # ENTRY + DOORS
    # --------------------------------------------------------

    def _pick_entry_room(self, layout) -> str:
        for r in layout:
            if self._is_living(r.name):
                return r.name
        for r in layout:
            if self._is_dining(r.name):
                return r.name

        public_rooms = [r for r in layout if r.room_type == "public"]
        if public_rooms:
            public_rooms.sort(key=lambda r: r.area, reverse=True)
            return public_rooms[0].name

        return sorted(layout, key=lambda r: r.area, reverse=True)[0].name

    def _place_main_door(self, room) -> Opening:
        width = 3.5
        margin = 2.0

        if self._touches_boundary_bottom(room) and room.w > width + 2 * margin:
            x = room.x + room.w * 0.18
            x = max(room.x + margin, min(x, room.x + room.w - width - margin))
            return Opening(x, -self.ext_t, width, self.ext_t, "door", "horizontal", "exterior", room.name, None, "main_door")

        if self._touches_boundary_left(room) and room.h > width + 2 * margin:
            y = room.y + room.h * 0.35
            y = max(room.y + margin, min(y, room.y + room.h - width - margin))
            return Opening(-self.ext_t, y, self.ext_t, width, "door", "vertical", "exterior", room.name, None, "main_door")

        if self._touches_boundary_right(room) and room.h > width + 2 * margin:
            y = room.y + room.h * 0.35
            y = max(room.y + margin, min(y, room.y + room.h - width - margin))
            return Opening(self.plot_width, y, self.ext_t, width, "door", "vertical", "exterior", room.name, None, "main_door")

        x = room.x + room.w * 0.50 - width / 2
        x = max(room.x + margin, min(x, room.x + room.w - width - margin))
        return Opening(x, self.plot_height, width, self.ext_t, "door", "horizontal", "exterior", room.name, None, "main_door")

    def _door_rule(self, room_a: str, room_b: str) -> Optional[Tuple[float, str]]:
        pair = {room_a, room_b}

        if any(self._is_living(x) for x in pair) and any(self._is_dining(x) for x in pair):
            return (5.0, "open_passage")
        if any(self._is_dining(x) for x in pair) and any(self._is_kitchen(x) for x in pair):
            return (4.5, "open_passage")
        if any(self._is_living(x) for x in pair) and any(self._is_kitchen(x) for x in pair):
            return (3.5, "standard")
        if any(self._is_living(x) for x in pair) and any(self._is_study(x) for x in pair):
            return (3.0, "standard")
        if self._is_bathroom(room_a) and self._is_bathroom(room_b):
            return None
        if (self._is_bathroom(room_a) and self._is_kitchen(room_b)) or (self._is_bathroom(room_b) and self._is_kitchen(room_a)):
            return None
        if (self._is_bathroom(room_a) and self._is_bedroom(room_b)) or (self._is_bathroom(room_b) and self._is_bedroom(room_a)):
            return (2.6, "standard")
        if self._is_bedroom(room_a) and self._is_bedroom(room_b):
            return (2.8, "standard")
        if (self._is_bedroom(room_a) and (self._is_living(room_b) or self._is_dining(room_b) or self._is_study(room_b))) or \
           (self._is_bedroom(room_b) and (self._is_living(room_a) or self._is_dining(room_a) or self._is_study(room_a))):
            return (3.0, "standard")

        return (3.0, "standard")

    def _place_adjacency_doors(self, layout, shared_map, adjacency_weights) -> List[Opening]:
        doors: List[Opening] = []
        used_pairs = set()

        scored_pairs = []
        for (a, b), weight in adjacency_weights.items():
            if (a, b) in shared_map:
                scored_pairs.append((weight, a, b, shared_map[(a, b)]))
            elif (b, a) in shared_map:
                scored_pairs.append((weight, a, b, shared_map[(b, a)]))

        scored_pairs.sort(reverse=True, key=lambda x: x[0])

        for weight, a, b, info in scored_pairs:
            pair_key = tuple(sorted((a, b)))
            if pair_key in used_pairs:
                continue

            rule = self._door_rule(a, b)
            if rule is None:
                continue

            width, style = rule
            op = self._make_opening_on_shared_wall(a, b, info, width, style)
            if op:
                doors.append(op)
                used_pairs.add(pair_key)

        return doors

    def _make_opening_on_shared_wall(self, room_a, room_b, info, opening_width, style) -> Optional[Opening]:
        margin = 1.5
        usable = info["shared_length"] - 2 * margin
        if usable <= 2.0:
            return None

        width = min(opening_width, usable)

        if info["orientation"] == "vertical":
            center_y = info["y"] + info["shared_length"] / 2
            y = center_y - width / 2
            return Opening(
                x=info["x"] - self.int_t / 2,
                y=y,
                w=self.int_t,
                h=width,
                opening_type="door",
                orientation="vertical",
                host_wall_type="interior",
                room_a=room_a,
                room_b=room_b,
                style=style,
            )
        else:
            center_x = info["x"] + info["shared_length"] / 2
            x = center_x - width / 2
            return Opening(
                x=x,
                y=info["y"] - self.int_t / 2,
                w=width,
                h=self.int_t,
                opening_type="door",
                orientation="horizontal",
                host_wall_type="interior",
                room_a=room_a,
                room_b=room_b,
                style=style,
            )

    def _door_graph(self, doors: List[Opening]) -> Dict[str, Set[str]]:
        graph = defaultdict(set)
        for d in doors:
            if d.room_a and d.room_b:
                graph[d.room_a].add(d.room_b)
                graph[d.room_b].add(d.room_a)
            elif d.room_a and d.style == "main_door":
                graph["OUTSIDE"].add(d.room_a)
                graph[d.room_a].add("OUTSIDE")
        return graph

    def _reachable_rooms(self, doors: List[Opening], entry_room: str) -> Set[str]:
        graph = self._door_graph(doors)
        visited = set()
        q = deque([entry_room])

        while q:
            cur = q.popleft()
            if cur in visited:
                continue
            visited.add(cur)
            for nxt in graph.get(cur, []):
                if nxt not in visited and nxt != "OUTSIDE":
                    q.append(nxt)
        return visited

    def _fallback_door_priority(self, a_name: str, b_name: str) -> float:
        pair = {a_name, b_name}

        if any(self._is_living(x) for x in pair) and any(self._is_dining(x) for x in pair):
            return 100
        if any(self._is_dining(x) for x in pair) and any(self._is_kitchen(x) for x in pair):
            return 95
        if any(self._is_bedroom(x) for x in pair) and any(self._is_bathroom(x) for x in pair):
            return 90
        if (self._is_bathroom(a_name) and self._is_kitchen(b_name)) or (self._is_bathroom(b_name) and self._is_kitchen(a_name)):
            return -100
        if self._is_bathroom(a_name) and self._is_bathroom(b_name):
            return -100
        if (self._is_bedroom(a_name) and (self._is_living(b_name) or self._is_dining(b_name))) or \
           (self._is_bedroom(b_name) and (self._is_living(a_name) or self._is_dining(a_name))):
            return 80
        if (self._is_study(a_name) and self._is_living(b_name)) or (self._is_study(b_name) and self._is_living(a_name)):
            return 78
        if self._is_bedroom(a_name) and self._is_bedroom(b_name):
            return 70
        return 50

    def _ensure_accessibility(self, layout, shared_map, doors: List[Opening], entry_room: str) -> List[Opening]:
        room_names = {r.name for r in layout}
        current_doors = list(doors)

        while True:
            reachable = self._reachable_rooms(current_doors, entry_room)
            unreachable = room_names - reachable
            if not unreachable:
                break

            best_candidate = None
            best_score = -1e9

            for room_u in unreachable:
                for (a, b), info in shared_map.items():
                    if room_u == a and b in reachable:
                        other = b
                    elif room_u == b and a in reachable:
                        other = a
                    else:
                        continue

                    if any(
                        ((d.room_a == room_u and d.room_b == other) or (d.room_a == other and d.room_b == room_u))
                        for d in current_doors if d.room_b is not None
                    ):
                        continue

                    rule = self._door_rule(room_u, other)
                    if rule is None:
                        continue

                    width, style = rule
                    priority = self._fallback_door_priority(room_u, other) + 0.2 * info["shared_length"]

                    if priority > best_score:
                        best_score = priority
                        best_candidate = (room_u, other, info, width, style)

            if best_candidate is None:
                break

            room_u, other, info, width, style = best_candidate
            op = self._make_opening_on_shared_wall(room_u, other, info, width, style)
            if op:
                current_doors.append(op)
            else:
                break

        return current_doors

    # --------------------------------------------------------
    # WINDOWS
    # --------------------------------------------------------

    def _window_rule(self, room_name: str, room_type: str) -> Tuple[int, float, str]:
        if self._is_living(room_name):
            return (2, 5.5, "window")
        if self._is_bedroom(room_name):
            return (1, 4.5, "window")
        if self._is_study(room_name):
            return (1, 4.0, "window")
        if self._is_kitchen(room_name):
            return (1, 3.5, "window")
        if self._is_bathroom(room_name):
            return (1, 2.0, "ventilator")
        if self._is_dining(room_name):
            return (1, 4.0, "window")
        return (1, 3.5, "window")

    def _place_windows(self, layout, existing_doors: List[Opening]) -> List[Opening]:
        windows: List[Opening] = []

        for room in layout:
            max_count, pref_width, style = self._window_rule(room.name, room.room_type)
            candidates = []

            if self._touches_boundary_left(room):
                candidates.append(("left", room.h))
            if self._touches_boundary_right(room):
                candidates.append(("right", room.h))
            if self._touches_boundary_bottom(room):
                candidates.append(("bottom", room.w))
            if self._touches_boundary_top(room):
                candidates.append(("top", room.w))

            candidates.sort(key=lambda x: x[1], reverse=True)
            count = 0

            for side, side_len in candidates:
                if count >= max_count:
                    break

                margin = 1.5
                usable = side_len - 2 * margin
                if usable <= 1.5:
                    continue

                win_w = min(pref_width, usable)
                trial_fracs = [0.5, 0.25, 0.75]
                placed = False

                for frac in trial_fracs:
                    if side == "left":
                        cy = room.y + room.h * frac
                        y = max(room.y + margin, min(cy - win_w / 2, room.y + room.h - win_w - margin))
                        candidate = Opening(-self.ext_t, y, self.ext_t, win_w, "window", "vertical", "exterior", room.name, None, style)
                    elif side == "right":
                        cy = room.y + room.h * frac
                        y = max(room.y + margin, min(cy - win_w / 2, room.y + room.h - win_w - margin))
                        candidate = Opening(self.plot_width, y, self.ext_t, win_w, "window", "vertical", "exterior", room.name, None, style)
                    elif side == "bottom":
                        cx = room.x + room.w * frac
                        x = max(room.x + margin, min(cx - win_w / 2, room.x + room.w - win_w - margin))
                        candidate = Opening(x, -self.ext_t, win_w, self.ext_t, "window", "horizontal", "exterior", room.name, None, style)
                    else:
                        cx = room.x + room.w * frac
                        x = max(room.x + margin, min(cx - win_w / 2, room.x + room.w - win_w - margin))
                        candidate = Opening(x, self.plot_height, win_w, self.ext_t, "window", "horizontal", "exterior", room.name, None, style)

                    conflict = False
                    for existing in existing_doors + windows:
                        if self._openings_conflict(candidate, existing, clearance=1.5):
                            conflict = True
                            break

                    if not conflict:
                        windows.append(candidate)
                        count += 1
                        placed = True
                        break

                if not placed:
                    continue

        return windows

    # --------------------------------------------------------
    # CLEARANCE ZONES
    # --------------------------------------------------------

    def _door_clearance_rects_for_room(self, room, doors: List[Opening]) -> List[Tuple[float, float, float, float]]:
        clearances = []

        for d in doors:
            belongs = (d.room_a == room.name or d.room_b == room.name or (d.style == "main_door" and d.room_a == room.name))
            if not belongs:
                continue

            if d.style == "main_door":
                depth = 3.5
                side_margin = 0.8
            elif d.style == "open_passage":
                depth = 2.5
                side_margin = 0.6
            else:
                depth = 2.4
                side_margin = 0.45

            if d.orientation == "vertical":
                local_depth = min(depth, room.w * 0.45)
                if abs(d.x + d.w - room.x) < 1.0 or abs(d.x - room.x) < 1.0:
                    rect = (room.x, d.y - side_margin, local_depth, d.h + 2 * side_margin)
                else:
                    rect = (room.x + room.w - local_depth, d.y - side_margin, local_depth, d.h + 2 * side_margin)
            else:
                local_depth = min(depth, room.h * 0.45)
                if abs(d.y + d.h - room.y) < 1.0 or abs(d.y - room.y) < 1.0:
                    rect = (d.x - side_margin, room.y, d.w + 2 * side_margin, local_depth)
                else:
                    rect = (d.x - side_margin, room.y + room.h - local_depth, d.w + 2 * side_margin, local_depth)

            clearances.append(rect)

        return clearances

    # --------------------------------------------------------
    # FURNITURE
    # --------------------------------------------------------

    def _candidate_ok(self, candidate_rect, forbidden_rects) -> bool:
        for fr in forbidden_rects:
            if self._rects_overlap(candidate_rect, fr):
                return False
        return True

    def _place_furniture(self, layout, doors: List[Opening]) -> List[FurnitureItem]:
        furniture: List[FurnitureItem] = []

        for room in layout:
            pad = 1.0
            forbidden = self._door_clearance_rects_for_room(room, doors)
            placed_rects = []

            def try_place(item_type, candidates):
                for cx, cy, cw, ch in candidates:
                    rect = (cx, cy, cw, ch)
                    room_bounds = (room.x + pad, room.y + pad, room.w - 2 * pad, room.h - 2 * pad)

                    if cx < room_bounds[0] or cy < room_bounds[1]:
                        continue
                    if cx + cw > room_bounds[0] + room_bounds[2]:
                        continue
                    if cy + ch > room_bounds[1] + room_bounds[3]:
                        continue

                    if not self._candidate_ok(rect, forbidden):
                        continue

                    if any(self._rects_overlap(rect, pr) for pr in placed_rects):
                        continue

                    furniture.append(FurnitureItem(room.name, item_type, cx, cy, cw, ch))
                    placed_rects.append(rect)
                    return True
                return False

            name = room.name

            # LIVING
            if self._is_living(name):
                sofa_sizes = [(7.0, 2.8), (6.0, 2.6), (5.0, 2.4)]
                placed = False
                for sw, sh in sofa_sizes:
                    sofa_candidates = [
                        (room.x + pad, room.y + pad, sw, sh),
                        (room.x + room.w - sw - pad, room.y + pad, sw, sh),
                        (room.x + pad, room.y + room.h - sh - pad, sw, sh),
                        (room.x + room.w - sw - pad, room.y + room.h - sh - pad, sw, sh),
                    ]
                    if try_place("Sofa", sofa_candidates):
                        placed = True
                        break

                if not placed:
                    try_place("Sofa", [
                        (room.x + pad, room.y + pad, 4.5, 2.2),
                    ])

                try_place("Center Tbl", [
                    (room.x + room.w * 0.40, room.y + room.h * 0.40, 3.0, 2.0),
                ])

            # BEDROOMS
            elif self._is_bedroom(name):
                bed_candidates = []
                standard_sizes = [(6.2, 5.6), (5.8, 5.2), (5.2, 4.8), (4.8, 4.4)]

                for bw, bh in standard_sizes:
                    bed_candidates.extend([
                        (room.x + pad, room.y + pad, bw, bh),
                        (room.x + room.w - bw - pad, room.y + pad, bw, bh),
                        (room.x + pad, room.y + room.h - bh - pad, bw, bh),
                        (room.x + room.w - bw - pad, room.y + room.h - bh - pad, bw, bh),
                        (room.x + pad, room.y + pad, bh, bw),
                        (room.x + room.w - bh - pad, room.y + pad, bh, bw),
                        (room.x + pad, room.y + room.h - bw - pad, bh, bw),
                        (room.x + room.w - bh - pad, room.y + room.h - bw - pad, bh, bw),
                    ])

                if not try_place("Bed", bed_candidates):
                    try_place("Bed", [
                        (room.x + pad, room.y + pad, 4.5, 4.0),
                    ])

                wr_sizes = [(4.0, 2.0), (3.5, 2.0), (3.0, 1.8)]
                wardrobe_candidates = []
                for ww, wh in wr_sizes:
                    wardrobe_candidates.extend([
                        (room.x + room.w - ww - pad, room.y + room.h - wh - pad, ww, wh),
                        (room.x + pad, room.y + room.h - wh - pad, ww, wh),
                        (room.x + room.w - ww - pad, room.y + pad, ww, wh),
                        (room.x + pad, room.y + pad, ww, wh),
                    ])
                try_place("Wardrobe", wardrobe_candidates)

            # DINING
            elif self._is_dining(name):
                dt_w = min(6.0, max(4.5, room.w * 0.34))
                dt_h = min(3.5, max(3.0, room.h * 0.22))
                try_place("Dining Tbl", [
                    (room.x + room.w * 0.55 - dt_w / 2, room.y + room.h * 0.28 - dt_h / 2, dt_w, dt_h),
                    (room.x + room.w * 0.55 - dt_w / 2, room.y + room.h * 0.44 - dt_h / 2, dt_w, dt_h),
                ])

            # KITCHEN
            elif self._is_kitchen(name):
                ct_h = 2.0
                ct_w = min(max(5.5, room.w - 2 * pad), room.w - 2 * pad)
                try_place("Counter", [
                    (room.x + pad, room.y + room.h - ct_h - pad, ct_w, ct_h),
                    (room.x + pad, room.y + pad, ct_w, ct_h),
                ])

            # STUDY
            elif self._is_study(name):
                desk_w = min(5.0, max(3.5, room.w * 0.35))
                desk_h = 2.2
                if not try_place("Desk", [
                    (room.x + pad, room.y + room.h - desk_h - pad, desk_w, desk_h),
                    (room.x + room.w - desk_w - pad, room.y + room.h - desk_h - pad, desk_w, desk_h),
                    (room.x + pad, room.y + pad, desk_w, desk_h),
                ]):
                    try_place("Desk", [
                        (room.x + pad, room.y + pad, 3.5, 2.0),
                    ])

            # BATHROOM
            elif self._is_bathroom(name):
                try_place("WC", [
                    (room.x + 0.8, room.y + 0.8, 2.0, 1.6),
                ])
                try_place("Sink", [
                    (room.x + room.w - 2.0 - 0.8, room.y + 0.8, 2.0, 1.2),
                ])

        return furniture

    # --------------------------------------------------------
    # LABELS
    # --------------------------------------------------------

    def _label_anchor(self, room) -> Tuple[float, float]:
        if self._is_living(room.name):
            return (room.x + room.w * 0.50, room.y + room.h * 0.66)
        if self._is_dining(room.name):
            return (room.x + room.w * 0.50, room.y + room.h * 0.70)
        if self._is_bedroom(room.name):
            return (room.x + room.w * 0.50, room.y + room.h * 0.78)
        if self._is_study(room.name):
            return (room.x + room.w * 0.56, room.y + room.h * 0.40)
        if self._is_kitchen(room.name):
            return (room.x + room.w * 0.55, room.y + room.h * 0.42)
        if self._is_bathroom(room.name):
            return (room.x + room.w * 0.52, room.y + room.h * 0.68)
        return (room.cx, room.cy)

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    def print_accessibility_report(self, scene: Dict) -> None:
        layout = scene["layout"]
        doors = scene["doors"]
        entry_room = scene["entry_room"]

        reachable = self._reachable_rooms(doors, entry_room)
        all_rooms = {r.name for r in layout}
        unreachable = all_rooms - reachable

        print("\n--- Accessibility Report ---")
        print(f"Entry room: {entry_room}")
        print(f"Total doors: {len(doors)}")
        print(f"Reachable rooms: {sorted(reachable)}")
        print(f"Unreachable rooms: {sorted(unreachable)}")
        if not unreachable:
            print("Status: ALL ROOMS ACCESSIBLE")
        else:
            print("Status: ACCESSIBILITY FAILURE")


# ============================================================
# 3. EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    requirements = [
        {"name": "Living Room", "room_type": "public", "min_area_sqft": 300, "adjacencies": ["Dining Area", "Study"]},
        {"name": "Dining Area", "room_type": "public", "min_area_sqft": 150, "adjacencies": ["Living Room", "Kitchen"]},
        {"name": "Bedroom 1", "room_type": "private", "min_area_sqft": 200, "adjacencies": ["Bathroom 1", "Bedroom 2"]},
        {"name": "Bedroom 2", "room_type": "private", "min_area_sqft": 150, "adjacencies": ["Bathroom 1"]},
        {"name": "Bedroom 3", "room_type": "private", "min_area_sqft": 140, "adjacencies": ["Bathroom 2"]},
        {"name": "Study", "room_type": "private", "min_area_sqft": 100, "adjacencies": ["Living Room"]},
        {"name": "Kitchen", "room_type": "service", "min_area_sqft": 120, "adjacencies": ["Dining Area"]},
        {"name": "Bathroom 1", "room_type": "service", "min_area_sqft": 55, "adjacencies": ["Bedroom 1", "Bedroom 2"]},
        {"name": "Bathroom 2", "room_type": "service", "min_area_sqft": 50, "adjacencies": ["Bedroom 3"]},
    ]

    total_area = 1500
    plot_width = 40

    phase2 = Phase2LayoutEngine(total_area=total_area, plot_width=plot_width, seed=42)
    best_layout, best_score, breakdown = phase2.generate_best_layout(
        requirements,
        num_candidates=220,
        verbose=True
    )

    plot_height = total_area / plot_width
    drafting = Phase3DraftingEngine(
        plot_width=plot_width,
        plot_height=plot_height,
        exterior_wall_thickness=0.75,
        interior_wall_thickness=0.375,
    )

    scene = drafting.build_drafting_scene(best_layout, requirements)
    drafting.print_accessibility_report(scene)
    drafting.visualize(scene, title="Phase 3 Drafting Engine Output")