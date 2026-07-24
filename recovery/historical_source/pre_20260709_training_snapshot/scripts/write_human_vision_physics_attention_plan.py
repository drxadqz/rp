from __future__ import annotations

from pathlib import Path


OUT = Path("reports/paper_protocol_summary/human_vision_physics_attention_plan.md")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(_markdown(), encoding="utf-8")
    print(OUT)


def _markdown() -> str:
    return """# Human-Vision And Physics-Attention Route

This note turns the human-eye / anti-human-eye idea into executable RSCD experiments. It is a method-design note, not a result claim.

## Motivation

Human drivers judge road slipperiness with useful but biased cues:

| human cue | useful part | weakness | algorithmic compensation |
|---|---|---|---|
| brightness / whiteness | snow and frost can be bright and low-saturation | overexposed concrete, glare, and snow can look similar | use snow-like mask plus texture and saturation checks, not brightness alone |
| shiny highlight | wet roads and water films can reflect light | small black-ice or thin-water regions may be missed | use specular and smooth-film pseudo masks with bottom-contact prior |
| color impression | wet asphalt often looks darker | camera exposure and shadow can fool humans | add dark-smooth water mask and compare with gradient/laplacian texture |
| global scene impression | fast coarse judgement | ignores small local slippery patches | use segmentation-style local pseudo masks and top-fraction pooling |
| central gaze bias | humans focus on obvious central/large regions | tire-contact risk depends on road-bottom/contact region | use vertical bottom prior and bottom-vs-top evidence differences |

## Implemented Candidate

`PhysicsAttentionBranch` implements weak segmentation-style physics attention without pixel labels.

It constructs soft pseudo masks from public RGB images:

- `snow_mask`: bright, low-saturation, snow/frost-like regions.
- `wet_highlight`: high-value, low-saturation specular regions.
- `dark_water`: dark and low-texture water/wet-asphalt regions.
- `smooth_film`: low-gradient, low-laplacian thin-film candidates.
- `rough_aggregate`: high-gradient/high-laplacian dry aggregate or roughness evidence.
- `wet_contact`: wet evidence multiplied by a bottom-contact prior.

For each mask the branch pools brightness, saturation, value, gradient, and laplacian evidence, plus entropy and local concentration. This mimics semantic segmentation attention but avoids needing manually annotated pixel masks.

## Anti-Human-Eye Claim Boundary

The claim is not that the model sees physical friction directly. The claim is narrower and testable:

- Human-like global RGB perception is vulnerable to glare, exposure, and snow/wet aliasing.
- The model explicitly checks local physical counter-cues: texture loss, laplacian smoothness, bottom-region wetness, dark smooth water, and high-frequency aggregate.
- If these cues improve RSCD hard slices or formal Macro-F1, the route is useful.
- If they hurt the same-split RSCD metrics or only increase complexity, prune or merge the useful masks into `PhysicsTexture`.

## Evaluation Gate

Fast-screen candidates:

- `fast_physics_attention_film`
- `fast_physics_attention_wavelet_film_gate_hier`

Promotion rule:

- Promote only if fast Top-1 or Macro-F1 beats both `fast_convnext_tiny` and `fast_physics_texture_quality` by more than 0.1 percentage point.
- If it only improves water/wet/ice slices, keep it as a targeted hard-slice module and require class-slice evidence.
- Otherwise prune it.

## Relation To Top-Venue Pattern

This route combines three publishable patterns:

1. Weak segmentation / query pooling: local evidence is pooled through pseudo masks.
2. Physics-guided attention: masks are built from optics and texture mechanisms rather than arbitrary learned attention.
3. Anti-shortcut design: bottom-contact and local texture statistics reduce reliance on dataset-level color/style shortcuts.
"""


if __name__ == "__main__":
    main()
