"""Hand-authored scene recipes: the offline cognitive director.

Why this exists
---------------
Judges hate a demo that dies with the Wi-Fi. The heuristic director produces a
*deliberately good* scene graph from a keyword match, so the full EEG -> attention ->
keyframe -> splat -> headset path runs with zero API keys. When ``MODEL_API_KEY`` is
present, Muse Spark replaces this and the rest of the pipeline does not change.

Layout values are authored, not guessed: each recipe is a plausible 512x512
photograph at standing eye height.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..schemas import BBox, Beat, SceneGraph, SceneObject


@dataclass
class Recipe:
    key: str
    title: str
    style: str
    lighting: str
    palette: list[str]
    keywords: tuple[str, ...]
    objects: list[tuple]  # (id, label, bbox, depth, salience, prompt, color)
    beats: list[tuple[str, list[str], str]] = field(default_factory=list)

    def build(self, narrative: str) -> SceneGraph:
        objects = [
            SceneObject(
                id=oid,
                label=label,
                bbox=BBox.from_xywh(*bbox),
                depth_hint=depth,
                base_salience=sal,
                prompt=prompt,
                color=color,
            )
            for oid, label, bbox, depth, sal, prompt, color in self.objects
        ]
        n = len(self.beats) or 1
        beats = [
            Beat(
                index=i,
                t0=round(i / n, 4),
                t1=round((i + 1) / n, 4),
                label=label,
                focus=focus,
                narrative=text,
            )
            for i, (label, focus, text) in enumerate(self.beats)
        ]
        return SceneGraph(
            title=self.title,
            narrative=narrative.strip(),
            style=self.style,
            lighting=self.lighting,
            palette=self.palette,
            objects=objects,
            beats=beats,
            source="heuristic",
            model=None,
        )


RECIPES: tuple[Recipe, ...] = (
    Recipe(
        key="study",
        title="The desk by the window",
        style="photorealistic memory, 35mm, shallow depth of field, slight halation",
        lighting="soft late-afternoon window light from the right, warm bounce",
        palette=["#2a2e36", "#7d8794", "#c8b49c", "#e6d9c6", "#f3efe7"],
        keywords=("desk", "study", "window", "office", "mug", "laptop", "work"),
        objects=[
            ("wall", "plaster wall", (0.00, 0.00, 1.00, 1.00), 0.98, 0.10,
             "warm off-white plaster wall, faint texture", "#d9d2c6"),
            ("window", "sash window", (0.50, 0.10, 0.42, 0.48), 0.88, 0.86,
             "tall sash window, sheer curtain, bright overcast sky beyond", "#cfdcea"),
            ("desk", "oak desk", (0.08, 0.62, 0.74, 0.26), 0.34, 0.72,
             "worn oak writing desk, visible grain, side light", "#8a6746"),
            ("mug", "ceramic mug", (0.29, 0.545, 0.105, 0.095), 0.26, 0.95,
             "small white ceramic mug, faint steam, thin handle", "#efeae1"),
            ("notebook", "open notebook", (0.45, 0.60, 0.17, 0.075), 0.30, 0.62,
             "open paper notebook with a pen resting in the gutter", "#f0ece1"),
            ("lamp", "brass lamp", (0.79, 0.44, 0.15, 0.30), 0.42, 0.44,
             "brass desk lamp with a warm bulb, thin stem", "#b08a4f"),
            ("plant", "fern", (0.01, 0.22, 0.13, 0.34), 0.55, 0.30,
             "small fern in a terracotta pot, soft focus", "#5c7350"),
            ("chair", "chair back", (0.02, 0.57, 0.22, 0.42), 0.12, 0.22,
             "wooden chair back in the foreground, out of focus", "#6b5237"),
        ],
        beats=[
            (
                "enter the room",
                ["window", "desk"],
                "The room is bright and the desk is where it should be.",
            ),
            ("the mug", ["mug", "desk"], "The mug is still warm. This is the part that stayed."),
            ("the page", ["notebook", "window"], "The page is open at the sentence that mattered."),
            ("leave", ["chair", "lamp"], "The chair is pushed back; the lamp is still on."),
        ],
    ),
    Recipe(
        key="kitchen",
        title="The kitchen at seven",
        style="photorealistic memory, 40mm, mild grain, warm highlights",
        lighting="low morning sun through a side window, steam in the air",
        palette=["#20242a", "#5d6873", "#b9a58c", "#dfd3c1", "#f6f2ea"],
        keywords=("kitchen", "kettle", "coffee", "stove", "breakfast", "counter"),
        objects=[
            ("wall", "tiled wall", (0.00, 0.00, 1.00, 1.00), 0.96, 0.10,
             "pale square tiles, grout lines, soft shadow", "#d7d9d4"),
            ("window", "kitchen window", (0.62, 0.08, 0.30, 0.34), 0.90, 0.70,
             "small kitchen window, bright morning sky, condensation", "#cfe0ea"),
            ("counter", "stone counter", (0.05, 0.60, 0.86, 0.25), 0.40, 0.55,
             "grey stone counter, clean edge, reflected light", "#9aa0a4"),
            ("kettle", "kettle", (0.24, 0.50, 0.16, 0.18), 0.30, 0.92,
             "stainless kettle just off the boil, thin steam", "#c3c8cc"),
            ("mug", "mug", (0.45, 0.545, 0.09, 0.10), 0.28, 0.80,
             "stoneware mug, dark rim, steam curling", "#e7e1d6"),
            ("bowl", "bowl", (0.62, 0.57, 0.13, 0.08), 0.35, 0.45,
             "shallow ceramic bowl with two lemons", "#e2d7b8"),
            ("shelf", "shelf", (0.03, 0.14, 0.34, 0.06), 0.66, 0.25,
             "wooden shelf with three jars, backlit", "#8b6f52"),
        ],
        beats=[
            (
                "first light",
                ["window", "counter"],
                "Light lands on the counter before anything else.",
            ),
            ("the kettle", ["kettle"], "The kettle is the loudest thing in the memory."),
            ("the mug", ["mug", "bowl"], "The mug is in my hand and it is too hot."),
            ("settle", ["shelf", "window"], "The room settles; the steam goes sideways."),
        ],
    ),
    Recipe(
        key="bedroom",
        title="The room with the low lamp",
        style="photorealistic memory, 50mm, soft bokeh, tungsten white balance",
        lighting="single warm lamp, deep falloff into the corners",
        palette=["#171a1f", "#3f4650", "#8d7460", "#cbb79c", "#eee5d6"],
        keywords=("bedroom", "bed", "sleep", "night", "lamp", "pillow", "book"),
        objects=[
            ("wall", "dark wall", (0.00, 0.00, 1.00, 1.00), 0.97, 0.12,
             "deep grey wall, soft tungsten gradient", "#4a4b4d"),
            ("bed", "bed", (0.08, 0.52, 0.76, 0.44), 0.30, 0.70,
             "unmade bed, linen duvet, soft folds", "#cfc5b6"),
            ("pillow", "pillow", (0.16, 0.46, 0.26, 0.13), 0.24, 0.78,
             "squashed pillow with a visible head dent", "#e6dfd2"),
            ("lamp", "bedside lamp", (0.78, 0.34, 0.13, 0.26), 0.38, 0.88,
             "small lamp with a linen shade, warm glow", "#e0b070"),
            ("book", "paperback", (0.66, 0.585, 0.13, 0.05), 0.34, 0.52,
             "paperback face down on the duvet, spine cracked", "#b9805e"),
            ("window", "curtained window", (0.02, 0.10, 0.22, 0.32), 0.86, 0.34,
             "window with heavy curtains, thin line of streetlight", "#7b8798"),
        ],
        beats=[
            ("lights off", ["lamp", "bed"], "The lamp is the last thing on."),
            ("the dent", ["pillow", "bed"], "The pillow still holds the shape of the night."),
            ("the book", ["book", "lamp"], "The book is face down, three pages in."),
            ("streetlight", ["window"], "A thin line of streetlight under the curtain."),
        ],
    ),
    Recipe(
        key="street",
        title="The corner after rain",
        style="photorealistic memory, 28mm, wet reflections, slight vignette",
        lighting="blue hour, sodium street lamps, wet asphalt",
        palette=["#10141b", "#2b3644", "#55677a", "#c08a4e", "#e8e2d4"],
        keywords=("street", "road", "city", "rain", "corner", "bench", "tree", "outside"),
        objects=[
            ("sky", "sky", (0.00, 0.00, 1.00, 0.42), 0.99, 0.10,
             "deep blue evening sky, thin cloud, first star", "#2b3a52"),
            ("road", "wet road", (0.00, 0.62, 1.00, 0.38), 0.80, 0.35,
             "wet asphalt with long reflected highlights", "#2c323a"),
            ("building", "building", (0.54, 0.06, 0.46, 0.62), 0.92, 0.45,
             "row of brick buildings, lit windows, soft edges", "#5a4a44"),
            ("tree", "tree", (0.03, 0.16, 0.34, 0.52), 0.70, 0.55,
             "bare tree, wet black branches against the sky", "#20242a"),
            ("bench", "bench", (0.30, 0.60, 0.30, 0.16), 0.45, 0.82,
             "painted iron bench, wet slats, one armrest", "#4a5a52"),
            ("lamp", "street lamp", (0.82, 0.10, 0.07, 0.66), 0.60, 0.74,
             "sodium street lamp with a halo in the damp air", "#e0a458"),
            ("puddle", "puddle", (0.12, 0.78, 0.28, 0.12), 0.74, 0.62,
             "shallow puddle mirroring the lamp", "#3d4b5c"),
        ],
        beats=[
            ("the corner", ["building", "road"], "The corner is empty in a way that felt staged."),
            ("under the lamp", ["lamp", "puddle"], "The lamp doubles itself in the puddle."),
            ("the bench", ["bench", "tree"], "The bench is wet; nobody is sitting on it."),
            ("walk on", ["road", "sky"], "I keep walking and the reflection breaks."),
        ],
    ),
)

GENERIC = Recipe(
    key="generic",
    title="A remembered room",
    style="photorealistic memory, 35mm, gentle grain",
    lighting="soft ambient daylight, no hard shadows",
    palette=["#262a31", "#69727d", "#a9967f", "#d8cbb9", "#f1ece2"],
    keywords=(),
    objects=[
        ("wall", "wall", (0.00, 0.00, 1.00, 1.00), 0.97, 0.12,
         "plain plaster wall, soft gradient", "#d5cfc4"),
        ("window", "window", (0.56, 0.12, 0.34, 0.40), 0.86, 0.68,
         "window with daylight behind it", "#cfdcea"),
        ("table", "table", (0.14, 0.58, 0.66, 0.28), 0.36, 0.62,
         "wooden table, worn edge", "#8a6b4d"),
        ("object", "object on the table", (0.34, 0.51, 0.12, 0.11), 0.30, 0.88,
         "small object on the table, sharp focus", "#e8e2d6"),
        ("chair", "chair", (0.02, 0.55, 0.20, 0.42), 0.14, 0.24,
         "chair back, foreground, out of focus", "#6b5237"),
    ],
    beats=[
        ("arrive", ["window", "table"], "The room registers as a whole first."),
        ("the object", ["object"], "One object carries almost all of the detail."),
        ("the edge", ["table", "chair"], "The room is already leaving the frame."),
    ],
)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", (text or "").lower()))


def pick_recipe(narrative: str) -> Recipe:
    toks = _tokens(narrative)
    best, best_score = GENERIC, 0
    for r in RECIPES:
        score = sum(2 if k in toks else 0 for k in r.keywords)
        if score > best_score:
            best, best_score = r, score
    return best


def heuristic_graph(narrative: str) -> SceneGraph:
    return pick_recipe(narrative).build(narrative)
