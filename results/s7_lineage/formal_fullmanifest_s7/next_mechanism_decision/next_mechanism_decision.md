# RSCD Next Mechanism Decision

- Candidate: `S7_full_baseline`
- Candidate dir: `E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_source_reliable_router_s7_20260709`
- Protocol: `full`
- Action: `design_next_custom_backbone`
- Mechanism target: `early_concrete_roughness_scale_space_expert`
- Promote full: `False`
- Full SOTA pass: `False`
- Reason: Dominant pressure is concrete slight/severe roughness, matching the measured RSCD failure mode.

## Metrics

| Metric | Candidate | Baseline | Delta |
|---|---:|---:|---:|
| top1 | 90.632% | 90.632% | +0.000 pp |
| macro_f1 | 88.920% | 88.920% | +0.000 pp |
| mean_precision | 88.729% | 88.729% | +0.000 pp |
| mean_recall | 89.226% | 89.226% | +0.000 pp |
| weighted_f1 | 90.654% | 90.654% | +0.000 pp |

## Top-1 SOTA Budget

- Full-test samples: `49500`
- Required extra correct predictions: `1103`

## Lowest-F1 Classes

| Class | F1 | P | R | Factor |
|---|---:|---:|---:|---|
| water_concrete_slight | 75.693% | 78.162% | 73.375% | water + concrete + slight |
| water_asphalt_slight | 79.271% | 72.803% | 87.000% | water + asphalt + slight |
| wet_concrete_slight | 80.384% | 77.278% | 83.750% | wet + concrete + slight |
| water_concrete_severe | 80.984% | 77.855% | 84.375% | water + concrete + severe |
| dry_concrete_slight | 83.247% | 84.042% | 82.468% | dry + concrete + slight |
| wet_concrete_severe | 83.515% | 85.771% | 81.375% | wet + concrete + severe |
| water_asphalt_severe | 83.916% | 85.382% | 82.500% | water + asphalt + severe |
| water_gravel | 85.990% | 83.178% | 89.000% | water + gravel + nonparam |
| dry_concrete_severe | 86.544% | 86.197% | 86.894% | dry + concrete + severe |
| dry_asphalt_severe | 86.595% | 82.393% | 91.250% | dry + asphalt + severe |

## Factor Pressure

### roughness

| True factor | Errors |
|---|---:|
| slight | 997 |
| severe | 720 |
| smooth | 480 |
| nonparam | 184 |
| ice | 30 |
| fresh | 27 |
| melted | 20 |

### friction

| True factor | Errors |
|---|---:|
| wet | 787 |
| water | 473 |
| dry | 348 |
| snow_ice | 17 |

### material_roughness

| True factor | Errors |
|---|---:|
| concrete_slight | 658 |
| concrete_severe | 512 |
| asphalt_slight | 412 |
| gravel_nonparam | 367 |
| asphalt_smooth | 361 |
| mud_nonparam | 330 |
| concrete_smooth | 301 |
| asphalt_severe | 267 |

### friction_material

| True factor | Errors |
|---|---:|
| wet_concrete | 406 |
| water_concrete | 384 |
| wet_asphalt | 348 |
| dry_concrete | 299 |
| dry_asphalt | 263 |
| wet_mud | 218 |
| water_asphalt | 210 |
| dry_gravel | 185 |

## Task-Adapted Mechanism Blueprint

- Candidate name hint: `S137_early_concrete_roughness_scale_space_expert`
- First-principle rationale: The dominant measured failure is not generic texture recognition; it is a concrete-conditioned roughness boundary. Concrete hides small height/texture changes under low-contrast wet/water film, so the backbone should sense multi-scale roughness before global pooling and condition that evidence on concrete likelihood.

### Targeted RSCD Factors

- material_roughness: concrete_slight and concrete_severe
- roughness axis: slight versus severe versus smooth
- secondary guard: water/wet concrete film should not be damaged

### Early/Mid Mechanism

- Compute gray/texture evidence at multiple scales inside the stem or stage-1 feature flow.
- Use gradient, Laplacian, and local-contrast energies after small/medium smoothing scales.
- Create a concrete-conditioned roughness gate from concrete proxy, wet/water film proxy, and scale-space texture contrast.
- Inject the gate as early FiLM/depthwise modulation into low-level feature maps, not as a late classifier head.
- Use separate gates for dry-concrete and wet/water-concrete because their visual coupling is different.

### Diagnostic Pairs

- `dry_concrete_slight -> dry_concrete_severe`
- `dry_concrete_severe -> dry_concrete_slight`
- `water_concrete_slight -> water_concrete_severe`
- `wet_concrete_severe -> wet_concrete_slight`

### Same-Budget Control

- Keep the same manifest caps, epochs, image size, batch size, and optimizer.
- Ablate only the claimed mechanism.
- Use a fixed or identity gate control when the mechanism is a learned gate.

### Screen Success Gate

- Top-1 must not decrease versus the same-budget screen baseline.
- Macro-F1 must not decrease versus the same-budget screen baseline.
- No key coupled class may drop by more than 0.5 pp F1.
- No non-key class may drop by more than 1.5 pp F1.

### Full Success Gate

- Full protocol must use the complete train/val/test manifests.
- Full-test Top-1 must reach at least 92.86%.
- Full-test Macro-F1 must reach at least 89.49%.
- Worst-class F1 and water_concrete_slight F1 must be reported explicitly.
