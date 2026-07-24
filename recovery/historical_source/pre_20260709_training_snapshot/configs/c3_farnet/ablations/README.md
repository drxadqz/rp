# C3-FaRNet ablation configs

Each YAML is self-contained and can be launched with:

```powershell
python train.py --config configs/c3_farnet/ablations/<file>.yaml
```

The configs share the same RSCD manifests, input size, seed, optimizer, 1-epoch screen budget, balanced sampling, and capped 120/class evaluation. They differ only in the ablated mechanism.

| File | Ablation target |
|---|---|
| `c3_ablation_01_flat_linear.yaml` | flat 27-class linear head |
| `c3_ablation_02_factor_only.yaml` | factor-only head |
| `c3_ablation_03_coupled_tensor.yaml` | coupled tensor head with Factor CE |
| `c3_ablation_04_no_triple.yaml` | coupled tensor without triple coupling |
| `c3_ablation_05_no_pairwise.yaml` | coupled tensor without pairwise interactions |
| `c3_ablation_06_no_factor_ce.yaml` | no Factor CE |
| `c3_ablation_07_no_tournament.yaml` | no Tournament Loss |
| `c3_ablation_08_no_counterfactual.yaml` | no Counterfactual Loss |
| `c3_ablation_09_no_reliability.yaml` | no Roughness Reliability loss |
| `c3_ablation_10_no_dryvor.yaml` | no DryConcreteVOR residual |
| `c3_ablation_11_late_physics_fusion_only.yaml` | late PhysicsTexture fusion only, without semantic/local physics branches |
| `c3_ablation_12_boundary_no_physics.yaml` | boundary experts without PhysicsTexture feature input; compare against `c3_farnet_full.yaml` where it is on |
