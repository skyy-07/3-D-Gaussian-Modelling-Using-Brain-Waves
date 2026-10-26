"""Prompt + JSON-schema contracts for the cognitive director (Muse Spark)."""

from __future__ import annotations

TEXT_MODEL = "muse-spark-1.3"

SYSTEM = """You are the spatial memory director for MnemoSplat, a research demo that turns a
spoken episodic memory into a 3D Gaussian-splat volume.

Your job: convert a first-person memory description into a *scene graph* that an image
model and a 3D lifting stage can both consume. You are not writing a story; you are
writing a layout sheet.

Hard rules
1. 4-8 objects. Each is a single concrete, drawable noun ("ceramic mug", "laptop",
   "sheer curtain"), never a person, face, or text.
2. Bounding boxes are normalised screen coordinates with origin at TOP-LEFT:
   u0,v0 = top-left corner, u1,v1 = bottom-right, all in [0,1].
   Objects must stay inside the frame, keep at least 0.04 width/height, and should not
   overlap each other by more than 30% of the smaller area.
3. Use the frame like a photograph at eye height. Floor-level objects sit low
   (v1 close to 1.0). Objects further away are smaller and sit higher.
4. depth_hint: 0.0 = arm's length, 1.0 = far wall / outside the window.
5. base_salience: how emotionally central the object is in the memory, 0..1. Only 1-2
   objects should exceed 0.8. This is a *prior*; measured EEG overrides it.
6. beats: 3-6 ordered moments of the memory. `focus` lists object ids that dominate that
   moment. Beat timing is assigned by the client - do not invent timestamps.
7. palette: 3-5 hex colours that describe the light in the memory.
8. style/lighting: short photographic descriptors, no brand names.

Never mention the person whose memory it is. Never output medical, diagnostic, or
identifying content. If the memory references a real identifiable person or private
location, describe only the generic setting.
"""

USER_TEMPLATE = """Memory transcript:
\"\"\"{narrative}\"\"\"

Additional constraints:
- The remembered viewpoint is {viewpoint}.
- Overall mood: {mood}.
- Target keyframe is {width}x{height} (aspect {aspect:.2f}).

Produce the scene graph as JSON matching the schema."""


def scene_graph_schema() -> dict:
    """OpenAI-compatible ``json_schema`` for the scene graph."""
    bbox = {
        "type": "object",
        "properties": {
            "u0": {"type": "number"},
            "v0": {"type": "number"},
            "u1": {"type": "number"},
            "v1": {"type": "number"},
        },
        "required": ["u0", "v0", "u1", "v1"],
        "additionalProperties": False,
    }
    obj = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "label": {"type": "string"},
            "bbox": bbox,
            "depth_hint": {"type": "number"},
            "base_salience": {"type": "number"},
            "prompt": {"type": "string"},
            "color": {"type": "string"},
        },
        "required": ["id", "label", "bbox", "depth_hint", "base_salience", "prompt"],
        "additionalProperties": False,
    }
    beat = {
        "type": "object",
        "properties": {
            "label": {"type": "string"},
            "focus": {"type": "array", "items": {"type": "string"}},
            "narrative": {"type": "string"},
        },
        "required": ["label", "focus"],
        "additionalProperties": False,
    }
    return {
        "name": "mnemosplat_scene_graph",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "style": {"type": "string"},
                "lighting": {"type": "string"},
                "palette": {"type": "array", "items": {"type": "string"}},
                "objects": {"type": "array", "items": obj, "minItems": 3, "maxItems": 8},
                "beats": {"type": "array", "items": beat, "minItems": 3, "maxItems": 6},
            },
            "required": ["title", "style", "lighting", "palette", "objects", "beats"],
            "additionalProperties": False,
        },
    }


def keyframe_prompt(graph, width: int, height: int, beat_label: str | None = None) -> str:
    """Image-model prompt. Deliberately camera-first and object-complete."""
    objects = ", ".join(
        f"{o.label} ({(o.bbox.u0 + o.bbox.u1) / 2:.2f}, {(o.bbox.v0 + o.bbox.v1) / 2:.2f})"
        for o in sorted(graph.objects, key=lambda o: o.depth_hint)
    )
    beat = f" Moment: {beat_label}." if beat_label else ""
    return (
        f"First-person photograph of a remembered room. {graph.style}. {graph.lighting}."
        f" Contents, back to front: {objects}.{beat}"
        " Natural perspective at standing eye height, no people, no text, no watermark."
        f" Photographic realism, subtle 35mm grain, {width}:{height} aspect."
    )
