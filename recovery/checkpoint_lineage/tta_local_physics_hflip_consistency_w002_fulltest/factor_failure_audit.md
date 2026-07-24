# RSCD Factor Failure Audit

- Run: `tta_local_physics_hflip_consistency_w002_fulltest`
- Samples: 49500
- Exact 27-class Top-1: 90.68%

## Factor-Level Metrics

| factor | accuracy | macro_f1 |
| --- | --- | --- |
| friction | 96.62% | 97.34% |
| material | 97.25% | 96.77% |
| roughness | 95.17% | 94.51% |

## Coupled Factor Accuracy

| factor_pair | accuracy |
| --- | --- |
| friction_material | 94.19% |
| friction_roughness | 92.25% |
| material_roughness | 93.50% |
| all_three | 90.68% |

## Error Factor Patterns

| wrong_factors | count | share_of_errors |
| --- | --- | --- |
| roughness | 1739 | 37.70% |
| friction | 1397 | 30.28% |
| material | 779 | 16.89% |
| material+roughness | 423 | 9.17% |
| friction+material+roughness | 116 | 2.51% |
| friction+roughness | 114 | 2.47% |
| friction+material | 45 | 0.98% |

## Lowest Exact Classes

| class | support | accuracy | friction_acc | material_acc | roughness_acc |
| --- | --- | --- | --- | --- | --- |
| water_concrete_slight | 800 | 75.50% | 92.50% | 96.00% | 83.50% |
| dry_concrete_slight | 2350 | 81.15% | 99.40% | 96.64% | 82.38% |
| water_concrete_severe | 800 | 81.88% | 97.12% | 95.25% | 84.62% |
| wet_concrete_severe | 800 | 82.38% | 94.38% | 97.12% | 86.75% |
| water_asphalt_severe | 800 | 83.25% | 96.88% | 94.12% | 87.38% |
| wet_concrete_slight | 800 | 83.62% | 94.00% | 97.62% | 88.88% |
| wet_asphalt_severe | 800 | 84.62% | 94.00% | 98.88% | 90.25% |
| water_asphalt_slight | 800 | 87.00% | 95.50% | 98.50% | 91.50% |
| dry_concrete_severe | 2350 | 87.06% | 98.77% | 98.09% | 88.77% |
| wet_concrete_smooth | 2350 | 87.28% | 89.40% | 98.64% | 98.38% |
| water_concrete_smooth | 2350 | 88.21% | 91.19% | 98.30% | 97.45% |
| water_gravel | 800 | 88.88% | 98.38% | 90.12% | 94.25% |

### friction top confusions

| true | pred | count |
| --- | --- | --- |
| wet | water | 562 |
| water | wet | 441 |
| dry | wet | 301 |
| wet | dry | 199 |
| fresh_snow | ice | 22 |
| dry | water | 18 |
| water | melted_snow | 18 |
| water | dry | 18 |
| ice | melted_snow | 17 |
| dry | fresh_snow | 13 |
| melted_snow | water | 11 |
| ice | fresh_snow | 11 |
| dry | ice | 11 |
| melted_snow | ice | 7 |
| fresh_snow | melted_snow | 7 |
| wet | melted_snow | 6 |
| ice | dry | 3 |
| dry | melted_snow | 2 |
| melted_snow | dry | 1 |
| water | ice | 1 |

### material top confusions

| true | pred | count |
| --- | --- | --- |
| mud | gravel | 272 |
| concrete | asphalt | 233 |
| gravel | mud | 232 |
| asphalt | concrete | 178 |
| concrete | gravel | 84 |
| gravel | concrete | 72 |
| asphalt | gravel | 64 |
| gravel | asphalt | 55 |
| concrete | mud | 43 |
| mud | concrete | 25 |
| asphalt | winter | 25 |
| concrete | winter | 22 |
| mud | asphalt | 20 |
| asphalt | mud | 16 |
| winter | asphalt | 11 |
| gravel | winter | 4 |
| winter | concrete | 3 |
| winter | mud | 2 |
| mud | winter | 2 |

### roughness top confusions

| true | pred | count |
| --- | --- | --- |
| slight | severe | 674 |
| severe | slight | 625 |
| smooth | slight | 369 |
| slight | smooth | 241 |
| slight | granular | 94 |
| severe | granular | 85 |
| granular | slight | 75 |
| granular | severe | 75 |
| smooth | winter | 38 |
| smooth | granular | 28 |
| granular | smooth | 22 |
| smooth | severe | 19 |
| severe | smooth | 16 |
| winter | smooth | 12 |
| granular | winter | 6 |
| slight | winter | 6 |
| severe | winter | 3 |
| winter | granular | 2 |
| winter | severe | 1 |
| winter | slight | 1 |
