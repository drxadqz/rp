from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from friction_affordance.ontology import MATERIALS, SNOW, UNEVENNESS, WETNESS, weak_mu_interval_from_state


CORE_STATE_TASKS = ("material", "wetness", "snow", "unevenness")


class FrictionSetHead(nn.Module):
    """Map a factorized road-state distribution to a friction affordance interval.

    The latent state is the Cartesian product of material, wetness, snow/ice, and
    unevenness labels. Each state has a weak physically motivated friction
    interval. The model predicts a distribution over states through the existing
    task heads, then marginalizes intervals under that distribution.
    """

    def __init__(self, entropy_expansion: float = 0.10) -> None:
        super().__init__()
        self.entropy_expansion = float(entropy_expansion)
        state_rows = []
        for material_idx, material in enumerate(MATERIALS):
            for wetness_idx, wetness in enumerate(WETNESS):
                for snow_idx, snow in enumerate(SNOW):
                    for unevenness_idx, _unevenness in enumerate(UNEVENNESS):
                        if snow != "none":
                            low, high = weak_mu_interval_from_state(snow=snow, material=material)
                        else:
                            low, high = weak_mu_interval_from_state(
                                friction=wetness,
                                wetness=wetness,
                                material=material,
                            )
                        if low is None or high is None:
                            low, high = 0.0, 1.2
                        state_rows.append((material_idx, wetness_idx, snow_idx, unevenness_idx, low, high))

        state = torch.tensor(state_rows, dtype=torch.float32)
        self.register_buffer("state_material_idx", state[:, 0].long())
        self.register_buffer("state_wetness_idx", state[:, 1].long())
        self.register_buffer("state_snow_idx", state[:, 2].long())
        self.register_buffer("state_unevenness_idx", state[:, 3].long())
        self.register_buffer("state_mu_low", state[:, 4])
        self.register_buffer("state_mu_high", state[:, 5])
        self.num_states = int(state.size(0))

    def forward(self, logits: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        log_material = F.log_softmax(logits["material"], dim=1)
        log_wetness = F.log_softmax(logits["wetness"], dim=1)
        log_snow = F.log_softmax(logits["snow"], dim=1)
        log_unevenness = F.log_softmax(logits["unevenness"], dim=1)

        state_log_prob = (
            log_material[:, self.state_material_idx]
            + log_wetness[:, self.state_wetness_idx]
            + log_snow[:, self.state_snow_idx]
            + log_unevenness[:, self.state_unevenness_idx]
        )
        # Product of normalized marginals over a full Cartesian state table should
        # already sum to one. The subtraction keeps AMP/numerical drift harmless.
        state_log_prob = state_log_prob - torch.logsumexp(state_log_prob, dim=1, keepdim=True)
        state_prob = state_log_prob.exp()

        low = torch.sum(state_prob * self.state_mu_low.view(1, -1), dim=1)
        high = torch.sum(state_prob * self.state_mu_high.view(1, -1), dim=1)
        entropy = -(state_prob * state_log_prob).sum(dim=1) / math.log(float(self.num_states))
        expansion = self.entropy_expansion * entropy
        interval = torch.stack(
            [
                (low - expansion).clamp(0.0, 1.2),
                (high + expansion).clamp(0.0, 1.2),
            ],
            dim=1,
        )
        mean = interval.mean(dim=1)
        z = 1.2815515655446004
        scale = ((interval[:, 1] - interval[:, 0]) / (2.0 * z)).clamp(1e-3, 1.0)
        return {
            "state_log_prob": state_log_prob,
            "state_prob": state_prob,
            "state_entropy": entropy,
            "mu_interval": interval,
            "mu_mean": mean,
            "mu_scale": scale,
            "state_material_idx": self.state_material_idx,
            "state_wetness_idx": self.state_wetness_idx,
            "state_snow_idx": self.state_snow_idx,
            "state_unevenness_idx": self.state_unevenness_idx,
        }
