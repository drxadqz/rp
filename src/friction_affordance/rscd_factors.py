from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


IGNORE_INDEX = -100

FRICTION_LABELS = ("dry", "wet", "water", "fresh_snow", "melted_snow", "ice")
MATERIAL_LABELS = ("none", "asphalt", "concrete", "mud", "gravel")
ROUGHNESS_LABELS = ("none", "smooth", "slight", "severe")
FACTOR_AXES = ("friction", "material", "roughness")
FACTOR_LABELS = {
    "friction": FRICTION_LABELS,
    "material": MATERIAL_LABELS,
    "roughness": ROUGHNESS_LABELS,
}


@dataclass(frozen=True)
class FactorTriple:
    friction: int
    material: int
    roughness: int

    def as_tuple(self) -> tuple[int, int, int]:
        return (int(self.friction), int(self.material), int(self.roughness))


@dataclass(frozen=True)
class HardPair:
    left: int
    right: int
    axis: str
    boundary: str


@dataclass(frozen=True)
class RSCDFactorSpec:
    class_to_idx: dict[str, int]
    class_to_factor: torch.Tensor
    valid_class_mask: torch.Tensor
    valid_factor_mask: torch.Tensor
    valid_tensor_mask: torch.Tensor
    class_index_grid: torch.Tensor
    hard_pairs: tuple[HardPair, ...]

    @property
    def num_classes(self) -> int:
        return int(self.class_to_factor.shape[0])


def canonical_class_label(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def factor_index(axis: str, value: str | None) -> int:
    labels = FACTOR_LABELS[axis]
    if value is None:
        value = "none"
    return labels.index(value) if value in labels else IGNORE_INDEX


def parse_rscd_label(name: str) -> FactorTriple:
    """Parse one RSCD class name into friction/material/roughness factors."""

    label = canonical_class_label(name)
    if label in {"fresh_snow", "melted_snow", "ice"}:
        return FactorTriple(
            factor_index("friction", label),
            factor_index("material", "none"),
            factor_index("roughness", "none"),
        )
    parts = label.split("_")
    friction = parts[0] if len(parts) >= 1 else None
    material = parts[1] if len(parts) >= 2 else "none"
    roughness = parts[2] if len(parts) >= 3 else "none"
    return FactorTriple(
        factor_index("friction", friction),
        factor_index("material", material),
        factor_index("roughness", roughness),
    )


def factor_name(axis: str, idx: int) -> str:
    if idx < 0:
        return "invalid"
    return FACTOR_LABELS[axis][int(idx)]


def boundary_name(axis: str, left: FactorTriple, right: FactorTriple) -> str:
    li = left.as_tuple()[FACTOR_AXES.index(axis)]
    ri = right.as_tuple()[FACTOR_AXES.index(axis)]
    a = factor_name(axis, li)
    b = factor_name(axis, ri)
    if axis == "friction" and {a, b} == {"wet", "water"}:
        return "wet_water"
    if axis == "friction" and {a, b} == {"dry", "wet"}:
        return "dry_wet"
    if axis == "friction" and {a, b} == {"fresh_snow", "melted_snow"}:
        return "snow_phase"
    if axis == "friction" and "ice" in {a, b}:
        return "snow_ice"
    if axis == "material" and {a, b} == {"asphalt", "concrete"}:
        return "asphalt_concrete"
    if axis == "material" and {a, b} == {"mud", "gravel"}:
        return "mud_gravel"
    if axis == "roughness" and {a, b} <= {"smooth", "slight", "severe"}:
        return "roughness"
    return f"{axis}_{a}_vs_{b}"


def build_rscd_factor_spec(class_to_idx: dict[str, int]) -> RSCDFactorSpec:
    """Build class-factor maps, valid combination mask, and hard-pair graph."""

    idx_to_class = {idx: canonical_class_label(name) for name, idx in class_to_idx.items()}
    num_classes = len(idx_to_class)
    class_to_factor = torch.full((num_classes, 3), IGNORE_INDEX, dtype=torch.long)
    valid_factor_mask = torch.zeros((num_classes, 3), dtype=torch.bool)
    class_index_grid = torch.full(
        (len(FRICTION_LABELS), len(MATERIAL_LABELS), len(ROUGHNESS_LABELS)),
        -1,
        dtype=torch.long,
    )
    triples: dict[int, FactorTriple] = {}
    for idx in range(num_classes):
        triple = parse_rscd_label(idx_to_class[idx])
        triples[idx] = triple
        values = torch.tensor(triple.as_tuple(), dtype=torch.long)
        class_to_factor[idx] = values
        valid_factor_mask[idx] = values.ge(0)
        if bool(values.ge(0).all()):
            class_index_grid[triple.friction, triple.material, triple.roughness] = int(idx)
    valid_tensor_mask = class_index_grid.ge(0)

    hard_pairs: list[HardPair] = []
    for i in range(num_classes):
        a = triples[i]
        av = a.as_tuple()
        if min(av) < 0:
            continue
        for j in range(i + 1, num_classes):
            b = triples[j]
            bv = b.as_tuple()
            if min(bv) < 0:
                continue
            diff_axes = [axis for axis, x, y in zip(FACTOR_AXES, av, bv, strict=True) if int(x) != int(y)]
            if len(diff_axes) != 1:
                continue
            axis = diff_axes[0]
            hard_pairs.append(HardPair(i, j, axis, boundary_name(axis, a, b)))

    return RSCDFactorSpec(
        class_to_idx={canonical_class_label(k): int(v) for k, v in class_to_idx.items()},
        class_to_factor=class_to_factor,
        valid_class_mask=class_to_factor.ge(0).all(dim=1),
        valid_factor_mask=valid_factor_mask,
        valid_tensor_mask=valid_tensor_mask,
        class_index_grid=class_index_grid,
        hard_pairs=tuple(hard_pairs),
    )


def class_factor_targets(labels: torch.Tensor, spec: RSCDFactorSpec, device: torch.device) -> dict[str, torch.Tensor]:
    factors = spec.class_to_factor.to(device=device).index_select(0, labels)
    return {
        "friction": factors[:, 0],
        "material": factors[:, 1],
        "roughness": factors[:, 2],
    }


def sanity_summary(class_to_idx: dict[str, int]) -> str:
    spec = build_rscd_factor_spec(class_to_idx)
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    lines = ["RSCD factor parsing sanity:"]
    for idx in range(spec.num_classes):
        f, m, r = spec.class_to_factor[idx].tolist()
        lines.append(
            f"{idx:02d} {idx_to_class[idx]} -> "
            f"({factor_name('friction', f)}, {factor_name('material', m)}, {factor_name('roughness', r)})"
        )
    by_axis: dict[str, int] = {axis: 0 for axis in FACTOR_AXES}
    by_boundary: dict[str, int] = {}
    for pair in spec.hard_pairs:
        by_axis[pair.axis] += 1
        by_boundary[pair.boundary] = by_boundary.get(pair.boundary, 0) + 1
    lines.append(f"hard_pairs={len(spec.hard_pairs)} by_axis={by_axis} by_boundary={by_boundary}")
    return "\n".join(lines)
