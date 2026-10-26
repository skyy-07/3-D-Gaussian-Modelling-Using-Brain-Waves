"""Built-in demo scenarios.

The focus/distracted pair deliberately shares **one** narrative and one scene graph.
Only the neural channel changes. That is the cleanest possible demo of the scientific
claim: same words, same objects, same camera - different memory density.
"""

from __future__ import annotations

from dataclasses import dataclass

STUDY = (
    "I am standing in the doorway of the study. The window is on the right, bright and "
    "cold, the sheer curtain pulled back. My desk is where it always was, oak, with the "
    "mug near the front edge, still warm. There is an open notebook with a pen in the "
    "gutter and the page turned down. The brass lamp is on even though it is daytime. "
    "I can see the chair pushed back and the fern on the side table by the window."
)

STREET = (
    "A corner after rain. The road is black and doubled by reflections. There is a bench "
    "under the tree, wet through, nobody sitting on it. The street lamp is already on and "
    "its halo sits in the puddle. The buildings are dark except for two windows. I keep "
    "walking and the reflection breaks."
)


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    narrative: str
    focus: tuple[float, ...]
    lateralization: tuple[float, ...]
    seed: int
    caption: str


SCENARIOS: dict[str, Scenario] = {
    "focus": Scenario(
        key="focus",
        label="Vivid recall",
        narrative=STUDY,
        # gamma/theta climbs, peaks around the mug + notebook, then falls away
        focus=(0.42, 0.66, 0.88, 0.96, 0.72, 0.40),
        lateralization=(-0.45, -0.15, 0.42, 0.62, 0.30, 0.02),
        seed=7,
        caption="High posterior gamma/theta, low spectral entropy: dense, sharp geometry.",
    ),
    "distracted": Scenario(
        key="distracted",
        label="Diffuse recall",
        narrative=STUDY,
        # same narrative, theta-dominated, effortful retrieval
        focus=(0.20, 0.16, 0.28, 0.24, 0.18, 0.14),
        lateralization=(0.10, -0.05, 0.12, -0.08, 0.04, -0.02),
        seed=21,
        caption="Theta-dominated, high entropy: the same room, but volumetric bokeh.",
    ),
    "street": Scenario(
        key="street",
        label="Blue hour",
        narrative=STREET,
        focus=(0.38, 0.58, 0.80, 0.62, 0.50, 0.30),
        lateralization=(0.35, 0.20, -0.30, -0.45, -0.20, 0.05),
        seed=33,
        caption="A second setting, to prove the pipeline is not tuned to one room.",
    ),
}


def get_scenario(key: str) -> Scenario:
    if key not in SCENARIOS:
        raise KeyError(f"unknown scenario {key!r}; choose from {sorted(SCENARIOS)}")
    return SCENARIOS[key]


def list_scenarios() -> list[dict]:
    return [
        {"key": s.key, "label": s.label, "caption": s.caption, "epochs": len(s.focus)}
        for s in SCENARIOS.values()
    ]
