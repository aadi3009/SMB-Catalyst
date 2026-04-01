import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ============================================================
# 1. DATA MODELS
# ============================================================

@dataclass
class RoomSpec:
    name: str
    room_type: str  # public / private / service / hybrid_bath
    min_area_sqft: float
    adjacencies: List[str] = field(default_factory=list)


@dataclass
class RoomRect:
    name: str
    room_type: str
    target_area: float
    x: float
    y: float
    w: float
    h: float
    zone_name: str

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def min_dim(self) -> float:
        return min(self.w, self.h)

    @property
    def max_dim(self) -> float:
        return max(self.w, self.h)

    @property
    def aspect_ratio(self) -> float:
        return self.max_dim / max(self.min_dim, 1e-6)


# ============================================================
# 2. LAYOUT ENGINE
# ============================================================

class Phase2LayoutEngine:
    def __init__(self, total_area: float, plot_width: float = 40.0, seed: int = 42):
        self.total_area = total_area
        self.plot_width = plot_width
        self.plot_height = total_area / plot_width
        self.rng = random.Random(seed)

        # Secondary/default adjacency rules
        self.default_adjacency_rules = {
            "Living Room": ["Dining Area", "Study Room"],
            "Dining Area": ["Living Room", "Kitchen"],
            "Kitchen": ["Dining Area"],
            "Master Bedroom": ["Bathroom 1", "Bedroom 2"],
            "Bedroom 2": ["Bathroom 1", "Bedroom 3", "Master Bedroom"],
            "Bedroom 3": ["Bathroom 2"],
            "Study Room": ["Living Room"],
            "Bathroom 1": ["Master Bedroom", "Bedroom 2"],
            "Bathroom 2": ["Bedroom 3"],
        }

        # Absolute minimum dimensions
        self.absolute_min_dim = {
            "public": 6.0,
            "private": 7.0,
            "service": 4.5,
            "hybrid_bath": 5.0,
        }

        # Preferred minimum dimensions
        self.preferred_min_dim = {
            "public": 8.0,
            "private": 9.0,
            "service": 5.0,
            "hybrid_bath": 5.5,
        }

        # Aspect ratio guidance
        self.ideal_aspect_upper = {
            "public": 2.4,
            "private": 2.2,
            "service": 2.4,
            "hybrid_bath": 2.0,
        }

        self.acceptable_aspect_upper = {
            "public": 3.2,
            "private": 3.0,
            "service": 3.0,
            "hybrid_bath": 2.6,
        }

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def generate_best_layout(
        self,
        room_requirements: List[Dict],
        num_candidates: int = 160,
        verbose: bool = True
    ) -> Tuple[List[RoomRect], float, Dict]:
        rooms = [self._parse_room(r) for r in room_requirements]
        adjacency_weights = self._build_adjacency_weights(rooms)

        candidates = []
        for i in range(num_candidates):
            candidate = self._generate_candidate(rooms, adjacency_weights, i)
            score, breakdown = self._score_layout(candidate, rooms, adjacency_weights)
            candidates.append((score, candidate, breakdown))

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_layout, best_breakdown = candidates[0]

        if verbose:
            print(f"\nGenerated {num_candidates} candidates")
            print(f"Best score: {best_score:.2f}")
            print("\n--- Best Layout Validation ---")
            self._print_validation(best_layout)
            print("\n--- Score Breakdown ---")
            for k, v in best_breakdown.items():
                print(f"{k:32s}: {v:8.2f}")

        return best_layout, best_score, best_breakdown

    def visualize(self, layout: List[RoomRect], title: Optional[str] = None) -> None:
        fig, ax = plt.subplots(figsize=(14, 9))
        ax.set_xlim(-1, self.plot_width + 1)
        ax.set_ylim(-1, self.plot_height + 1)
        ax.set_aspect("equal")

        zone_colors = {
            "public": "#d9edf7",
            "private": "#dff0d8",
            "service": "#fcf8e3",
            "hybrid_bath": "#f5efe1",
        }

        outer = patches.Rectangle(
            (0, 0), self.plot_width, self.plot_height,
            linewidth=3, edgecolor="black", facecolor="none"
        )
        ax.add_patch(outer)

        for rect in layout:
            patch = patches.Rectangle(
                (rect.x, rect.y), rect.w, rect.h,
                linewidth=2, edgecolor="black",
                facecolor=zone_colors.get(rect.room_type, "whitesmoke"),
                alpha=0.9
            )
            ax.add_patch(patch)

            label = (
                f"{rect.name}\n"
                f"{int(round(rect.area))} sqft\n"
                f"{rect.w:.1f}' x {rect.h:.1f}'"
            )
            ax.text(
                rect.cx, rect.cy, label,
                ha="center", va="center",
                fontsize=10, weight="bold"
            )

        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_xlabel("Feet", fontsize=14)
        ax.set_ylabel("Feet", fontsize=14)
        ax.set_title(
            title or f"Revised First-Pass Architectural Layout ({int(self.total_area)} sq ft)",
            fontsize=22
        )
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------

    def _parse_room(self, r: Dict) -> RoomSpec:
        name = r["name"]
        area = float(r["min_area_sqft"])
        room_type = self._normalize_room_type(r.get("room_type", ""), name)
        adj = list(r.get("adjacencies", []))
        return RoomSpec(
            name=name,
            room_type=room_type,
            min_area_sqft=area,
            adjacencies=adj
        )

    def _normalize_room_type(self, room_type: str, name: str) -> str:
        rt = room_type.strip().lower()
        nl = name.lower()

        if "bathroom" in nl or "toilet" in nl or "wc" in nl:
            return "hybrid_bath"

        if rt in {"public", "private", "service", "hybrid_bath"}:
            return rt

        if any(k in nl for k in ["living", "dining", "foyer", "lounge"]):
            return "public"
        if any(k in nl for k in ["bedroom", "study", "master"]):
            return "private"
        return "service"

    # ------------------------------------------------------------
    # Adjacency weights
    # ------------------------------------------------------------

    def _build_adjacency_weights(self, rooms: List[RoomSpec]) -> Dict[Tuple[str, str], float]:
        names = {r.name for r in rooms}
        weights: Dict[Tuple[str, str], float] = {}

        def add_weight(a: str, b: str, w: float):
            if a == b:
                return
            key = tuple(sorted((a, b)))
            weights[key] = max(weights.get(key, 0.0), w)

        # Primary priority: customer-defined adjacency
        for room in rooms:
            for other in room.adjacencies:
                if other in names:
                    add_weight(room.name, other, 14.0)

        # Secondary: inferred/default rules
        for room in rooms:
            for other in self.default_adjacency_rules.get(room.name, []):
                if other in names:
                    add_weight(room.name, other, 5.0)

        # Mild fallback clustering
        for i in range(len(rooms)):
            for j in range(i + 1, len(rooms)):
                a = rooms[i]
                b = rooms[j]
                key = tuple(sorted((a.name, b.name)))
                if key in weights:
                    continue

                # bedrooms cluster mildly
                if a.room_type == "private" and b.room_type == "private":
                    weights[key] = 1.4

                # bathrooms mildly relate to private zone
                elif a.room_type == "hybrid_bath" and b.room_type == "private":
                    weights[key] = 1.8
                elif b.room_type == "hybrid_bath" and a.room_type == "private":
                    weights[key] = 1.8

        return weights

    # ------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------

    def _generate_candidate(
        self,
        rooms: List[RoomSpec],
        adjacency_weights: Dict[Tuple[str, str], float],
        candidate_idx: int
    ) -> List[RoomRect]:
        public_rooms = [r for r in rooms if r.room_type == "public"]
        private_rooms = [r for r in rooms if r.room_type == "private"]
        service_rooms = [r for r in rooms if r.room_type == "service"]
        hybrid_baths = [r for r in rooms if r.room_type == "hybrid_bath"]

        # Randomly assign bathrooms to private or service for each candidate
        private_group = private_rooms[:]
        service_group = service_rooms[:]

        for bath in hybrid_baths:
            if self.rng.random() < 0.55:
                private_group.append(bath)
            else:
                service_group.append(bath)

        zones = self._sample_macro_zones(public_rooms, private_group, service_group)

        ordered_public = self._order_rooms(public_rooms, adjacency_weights, candidate_idx + 10)
        ordered_private = self._order_rooms(private_group, adjacency_weights, candidate_idx + 1000)
        ordered_service = self._order_rooms(service_group, adjacency_weights, candidate_idx + 2000)

        rects = []
        rects.extend(self._layout_zone(zones["public"], ordered_public, "public"))
        rects.extend(self._layout_zone(zones["private"], ordered_private, "private"))
        rects.extend(self._layout_zone(zones["service"], ordered_service, "service"))

        return rects

    def _sample_macro_zones(
        self,
        public_rooms: List[RoomSpec],
        private_rooms: List[RoomSpec],
        service_rooms: List[RoomSpec]
    ) -> Dict[str, Tuple[float, float, float, float]]:
        public_area = sum(r.min_area_sqft for r in public_rooms)
        private_area = sum(r.min_area_sqft for r in private_rooms)
        service_area = sum(r.min_area_sqft for r in service_rooms)
        total = max(public_area + private_area + service_area, 1e-6)

        public_h = self.plot_height * (public_area / total)
        public_h = max(self.plot_height * 0.18, min(public_h, self.plot_height * 0.42))
        public_h *= self.rng.uniform(0.88, 1.12)
        public_h = max(self.plot_height * 0.16, min(public_h, self.plot_height * 0.45))

        upper_h = self.plot_height - public_h
        upper_total = max(private_area + service_area, 1e-6)

        private_w = self.plot_width * (private_area / upper_total)
        private_w = max(self.plot_width * 0.38, min(private_w, self.plot_width * 0.82))
        private_w *= self.rng.uniform(0.85, 1.15)
        private_w = max(self.plot_width * 0.35, min(private_w, self.plot_width * 0.85))

        service_w = self.plot_width - private_w

        return {
            "public": (0.0, 0.0, self.plot_width, public_h),
            "private": (0.0, public_h, private_w, upper_h),
            "service": (private_w, public_h, service_w, upper_h),
        }

    def _order_rooms(
        self,
        rooms: List[RoomSpec],
        adjacency_weights: Dict[Tuple[str, str], float],
        salt: int
    ) -> List[RoomSpec]:
        if not rooms:
            return []

        local_rng = random.Random(salt)
        remaining = rooms[:]

        connectivity = {}
        for r in remaining:
            conn = 0.0
            for s in remaining:
                if r.name == s.name:
                    continue
                conn += adjacency_weights.get(tuple(sorted((r.name, s.name))), 0.0)
            connectivity[r.name] = conn

        if local_rng.random() < 0.5:
            remaining.sort(key=lambda r: (connectivity[r.name], r.min_area_sqft), reverse=True)
        else:
            remaining.sort(key=lambda r: (r.min_area_sqft, connectivity[r.name]), reverse=True)

        ordered = [remaining.pop(0)]

        while remaining:
            scored = []
            for r in remaining:
                adj_score = sum(adjacency_weights.get(tuple(sorted((r.name, o.name))), 0.0) for o in ordered)
                size_score = r.min_area_sqft / 1200.0
                jitter = local_rng.uniform(-0.15, 0.15)
                scored.append((adj_score + size_score + jitter, r))
            scored.sort(key=lambda x: x[0], reverse=True)
            chosen = scored[0][1]
            ordered.append(chosen)
            remaining.remove(chosen)

        # diversify a bit
        for i in range(len(ordered) - 1):
            if local_rng.random() < 0.15:
                ordered[i], ordered[i + 1] = ordered[i + 1], ordered[i]

        return ordered

    # ------------------------------------------------------------
    # Layout inside zone
    # ------------------------------------------------------------

    def _layout_zone(self, zone_rect: Tuple[float, float, float, float], rooms: List[RoomSpec], zone_name: str) -> List[RoomRect]:
        if not rooms:
            return []
        x, y, w, h = zone_rect
        return self._recursive_partition(x, y, w, h, rooms, zone_name)

    def _recursive_partition(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        rooms: List[RoomSpec],
        zone_name: str
    ) -> List[RoomRect]:
        if len(rooms) == 1:
            r = rooms[0]
            return [RoomRect(r.name, r.room_type, r.min_area_sqft, x, y, w, h, zone_name)]

        best = None
        total_area = sum(r.min_area_sqft for r in rooms)

        for split_idx in range(1, len(rooms)):
            group1 = rooms[:split_idx]
            group2 = rooms[split_idx:]

            a1 = sum(r.min_area_sqft for r in group1)
            ratio1 = a1 / total_area

            for orientation in ["vertical", "horizontal"]:
                if orientation == "vertical":
                    w1 = w * ratio1
                    w2 = w - w1
                    if w1 <= 0 or w2 <= 0:
                        continue
                    rects1 = self._recursive_partition(x, y, w1, h, group1, zone_name)
                    rects2 = self._recursive_partition(x + w1, y, w2, h, group2, zone_name)
                else:
                    h1 = h * ratio1
                    h2 = h - h1
                    if h1 <= 0 or h2 <= 0:
                        continue
                    rects1 = self._recursive_partition(x, y, w, h1, group1, zone_name)
                    rects2 = self._recursive_partition(x, y + h1, w, h2, group2, zone_name)

                candidate = rects1 + rects2
                penalty = self._shape_penalty(candidate) + 0.8 * self._min_dim_penalty(candidate)
                balance = -abs((w if orientation == "vertical" else h) * ratio1 - (w if orientation == "vertical" else h) * (1 - ratio1))
                score = -penalty + 0.05 * balance

                if best is None or score > best[0]:
                    best = (score, candidate)

        if best is None:
            return self._fallback_slice(x, y, w, h, rooms, zone_name)

        return best[1]

    def _fallback_slice(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        rooms: List[RoomSpec],
        zone_name: str
    ) -> List[RoomRect]:
        total = sum(r.min_area_sqft for r in rooms)
        rects = []
        cur_x = x
        for i, r in enumerate(rooms):
            if i == len(rooms) - 1:
                rw = x + w - cur_x
            else:
                rw = w * (r.min_area_sqft / total)
            rects.append(RoomRect(r.name, r.room_type, r.min_area_sqft, cur_x, y, rw, h, zone_name))
            cur_x += rw
        return rects

    # ------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------

    def _score_layout(
        self,
        layout: List[RoomRect],
        room_specs: List[RoomSpec],
        adjacency_weights: Dict[Tuple[str, str], float]
    ) -> Tuple[float, Dict]:
        by_name = {r.name: r for r in layout}

        breakdown = {
            "adjacency_reward": 0.0,
            "distance_reward": 0.0,
            "shape_penalty": 0.0,
            "min_dim_penalty": 0.0,
            "zone_penalty": 0.0,
            "exterior_reward": 0.0,
            "plumbing_reward": 0.0,
            "private_bath_reward": 0.0,
            "area_penalty": 0.0,
        }

        for (a, b), weight in adjacency_weights.items():
            if a not in by_name or b not in by_name:
                continue

            ra = by_name[a]
            rb = by_name[b]

            shared_len = self._shared_edge_length(ra, rb)
            point_touch = self._point_touch_only(ra, rb)
            dist = self._center_distance(ra, rb)

            if shared_len > 0.1:
                breakdown["adjacency_reward"] += weight * (8.0 + 0.5 * shared_len)
            elif point_touch:
                breakdown["adjacency_reward"] += weight * 0.8
            else:
                breakdown["distance_reward"] += max(0.0, weight * (4.5 - 0.16 * dist))

        breakdown["shape_penalty"] = self._shape_penalty(layout)
        breakdown["min_dim_penalty"] = self._min_dim_penalty(layout)
        breakdown["zone_penalty"] = self._zone_penalty(layout)
        breakdown["exterior_reward"] = self._exterior_wall_reward(layout)
        breakdown["plumbing_reward"] = self._plumbing_reward(layout)
        breakdown["private_bath_reward"] = self._private_bath_reward(layout)

        for spec in room_specs:
            rect = by_name[spec.name]
            breakdown["area_penalty"] += abs(rect.area - spec.min_area_sqft) * 0.12

        total_score = (
            breakdown["adjacency_reward"]
            + breakdown["distance_reward"]
            + breakdown["exterior_reward"]
            + breakdown["plumbing_reward"]
            + breakdown["private_bath_reward"]
            - breakdown["shape_penalty"]
            - breakdown["min_dim_penalty"]
            - breakdown["zone_penalty"]
            - breakdown["area_penalty"]
        )
        return total_score, breakdown

    def _shape_penalty(self, layout: List[RoomRect]) -> float:
        penalty = 0.0
        for r in layout:
            ar = r.aspect_ratio
            ideal = self.ideal_aspect_upper[r.room_type]
            acceptable = self.acceptable_aspect_upper[r.room_type]

            if ar <= ideal:
                continue
            elif ar <= acceptable:
                penalty += 10.0 * (ar - ideal)
            else:
                penalty += 10.0 * (acceptable - ideal) + 40.0 * (ar - acceptable)
        return penalty

    def _min_dim_penalty(self, layout: List[RoomRect]) -> float:
        penalty = 0.0
        for r in layout:
            abs_min = self.absolute_min_dim[r.room_type]
            pref_min = self.preferred_min_dim[r.room_type]

            if r.min_dim < abs_min:
                penalty += 320.0 + 140.0 * (abs_min - r.min_dim)
            elif r.min_dim < pref_min:
                penalty += 25.0 * (pref_min - r.min_dim)
        return penalty

    def _zone_penalty(self, layout: List[RoomRect]) -> float:
        """
        Bathrooms can sit in private OR service without penalty.
        """
        penalty = 0.0
        for r in layout:
            x_norm = r.cx / self.plot_width
            y_norm = r.cy / self.plot_height

            if r.room_type == "public":
                penalty += max(0.0, (y_norm - 0.45)) * 18.0

            elif r.room_type == "private":
                penalty += max(0.0, (0.25 - y_norm)) * 16.0

            elif r.room_type == "service":
                penalty += max(0.0, (0.52 - x_norm)) * 8.0
                penalty += max(0.0, (0.34 - y_norm)) * 8.0

            elif r.room_type == "hybrid_bath":
                # no penalty in private or service zones
                pass

        return penalty

    def _exterior_wall_reward(self, layout: List[RoomRect]) -> float:
        reward = 0.0
        for r in layout:
            if not self._touches_outer_boundary(r):
                continue

            if r.name in {"Living Room", "Master Bedroom", "Bedroom 2", "Bedroom 3", "Study Room"}:
                reward += 8.0
            elif r.name in {"Kitchen", "Dining Area"}:
                reward += 4.0
            elif "Bathroom" in r.name:
                reward += 2.0
        return reward

    def _plumbing_reward(self, layout: List[RoomRect]) -> float:
        """
        Reduced strength versus previous version.
        Kitchen near bathrooms is good, but not dominant.
        """
        by_name = {r.name: r for r in layout}
        reward = 0.0

        kitchen = by_name.get("Kitchen")
        bath1 = by_name.get("Bathroom 1")
        bath2 = by_name.get("Bathroom 2")

        if kitchen and bath1:
            reward += max(0.0, 5.0 - 0.20 * self._center_distance(kitchen, bath1))
        if kitchen and bath2:
            reward += max(0.0, 5.0 - 0.20 * self._center_distance(kitchen, bath2))
        if bath1 and bath2:
            reward += max(0.0, 4.0 - 0.18 * self._center_distance(bath1, bath2))

        return reward

    def _private_bath_reward(self, layout: List[RoomRect]) -> float:
        """
        Stronger reward for bathrooms being near bedrooms,
        especially master bedroom.
        """
        by_name = {r.name: r for r in layout}
        reward = 0.0

        preferred_pairs = [
            ("Master Bedroom", "Bathroom 1", 12.0),
            ("Bedroom 2", "Bathroom 1", 8.0),
            ("Bedroom 3", "Bathroom 2", 8.0),
        ]

        for a, b, base in preferred_pairs:
            if a in by_name and b in by_name:
                ra = by_name[a]
                rb = by_name[b]
                shared_len = self._shared_edge_length(ra, rb)
                dist = self._center_distance(ra, rb)

                if shared_len > 0.1:
                    reward += base + 0.5 * shared_len
                else:
                    reward += max(0.0, base - 0.25 * dist)

        return reward

    # ------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------

    def _center_distance(self, a: RoomRect, b: RoomRect) -> float:
        return math.hypot(a.cx - b.cx, a.cy - b.cy)

    def _interval_overlap(self, a1: float, a2: float, b1: float, b2: float) -> float:
        return max(0.0, min(a2, b2) - max(a1, b1))

    def _shared_edge_length(self, a: RoomRect, b: RoomRect) -> float:
        eps = 1e-6

        if abs((a.x + a.w) - b.x) < eps or abs((b.x + b.w) - a.x) < eps:
            return self._interval_overlap(a.y, a.y + a.h, b.y, b.y + b.h)

        if abs((a.y + a.h) - b.y) < eps or abs((b.y + b.h) - a.y) < eps:
            return self._interval_overlap(a.x, a.x + a.w, b.x, b.x + b.w)

        return 0.0

    def _point_touch_only(self, a: RoomRect, b: RoomRect) -> bool:
        eps = 1e-6
        corners_a = [
            (a.x, a.y),
            (a.x + a.w, a.y),
            (a.x, a.y + a.h),
            (a.x + a.w, a.y + a.h),
        ]
        corners_b = [
            (b.x, b.y),
            (b.x + b.w, b.y),
            (b.x, b.y + b.h),
            (b.x + b.w, b.y + b.h),
        ]
        for ax, ay in corners_a:
            for bx, by in corners_b:
                if abs(ax - bx) < eps and abs(ay - by) < eps:
                    return self._shared_edge_length(a, b) < eps
        return False

    def _touches_outer_boundary(self, r: RoomRect) -> bool:
        eps = 1e-6
        return (
            abs(r.x - 0.0) < eps
            or abs(r.y - 0.0) < eps
            or abs((r.x + r.w) - self.plot_width) < eps
            or abs((r.y + r.h) - self.plot_height) < eps
        )

    # ------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------

    def _print_validation(self, layout: List[RoomRect]) -> None:
        for r in layout:
            abs_min = self.absolute_min_dim[r.room_type]
            pref_min = self.preferred_min_dim[r.room_type]
            ideal_ar = self.ideal_aspect_upper[r.room_type]
            acceptable_ar = self.acceptable_aspect_upper[r.room_type]

            dim_status = (
                "ABS_FAIL" if r.min_dim < abs_min
                else "PREF_FAIL" if r.min_dim < pref_min
                else "OK"
            )

            ar_status = (
                "OK" if r.aspect_ratio <= ideal_ar
                else "WARN" if r.aspect_ratio <= acceptable_ar
                else "FAIL"
            )

            print(
                f"{r.name:15s} | {r.room_type:11s} | zone={r.zone_name:7s} | "
                f"{r.w:6.2f} x {r.h:6.2f} | "
                f"actual={r.area:7.1f} | target={r.target_area:7.1f} | "
                f"AR={r.aspect_ratio:4.2f} | min_dim={r.min_dim:5.2f} | "
                f"{dim_status}, {ar_status}"
            )


# ============================================================
# 3. EXAMPLE USAGE
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




    engine = Phase2LayoutEngine(total_area=1500, plot_width=40, seed=42)
    best_layout, best_score, breakdown = engine.generate_best_layout(
        requirements,
        num_candidates=220,
        verbose=True
    )
    engine.visualize(best_layout)