"""
UAFCP-Net (lightweight)

3D U-Net backbone + uncertainty-guided conservative logit refinement.

Key features:
- Backbone-agnostic refinement module
- Uncertainty-gated conservative propagation in logit space
- Edge-aware modulation in decision-aligned feature space

Designed for anonymous open-source release.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import UNet


# =========================================================
# Main Network
# =========================================================

class UAFCPNet(nn.Module):
    """
    UAFCP-Net: 3D UNet + uncertainty-guided conservative refinement.

    Args:
        in_channels: input image channels
        feat_channels: backbone output feature channels
        use_uccp: enable refinement module
    """

    def __init__(
        self,
        in_channels=1,
        feat_channels=64,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="INSTANCE",
        act="RELU",
        use_uccp=True,
        uccp_steps=2,
        uccp_eta=0.3,
        uccp_u0=0.5,
        uccp_tau=0.1,
        uccp_source_term=True,
    ):
        super().__init__()

        self.use_uccp = use_uccp

        # backbone
        self.backbone = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feat_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            norm=norm,
            act=act,
        )

        # logit + feature projection
        self.logit_head = nn.Conv3d(feat_channels, 2, kernel_size=1)
        self.feat_head = nn.Conv3d(feat_channels, 2, kernel_size=1)

        # refinement module
        if self.use_uccp:
            self.refine = UQFluxRefine(
                K_steps=uccp_steps,
                eta=uccp_eta,
                u0=uccp_u0,
                tau=uccp_tau,
                source_term=uccp_source_term,
                feat_channels=2,
            )

    def forward(self, x):
        feat_backbone = self.backbone(x)
        feat = self.feat_head(feat_backbone)
        logits = self.logit_head(feat_backbone)

        if self.use_uccp:
            logits = self.refine(logits, feat)

        return logits


# =========================================================
# UQ-Guided Conservative Refinement
# =========================================================

class UQFluxRefine(nn.Module):
    """
    Uncertainty-gated conservative logit refinement.

    Operates in low-dimensional decision-aligned feature space.
    """

    def __init__(
        self,
        K_steps=2,
        eta=0.3,
        u0=0.5,
        tau=0.1,
        source_term=True,
        feat_channels=2,
    ):
        super().__init__()

        self.K_steps = K_steps
        self.eta = eta
        self.u0 = u0
        self.tau = tau
        self.use_source = source_term

        self.stencil = DepthwiseStencil3D(1)
        self.stencil_feat = DepthwiseStencil3D(feat_channels)
        self.edge_mlp = EdgeMLP(feat_channels)

    def forward(self, logits0, feat):
        logits = logits0

        for _ in range(self.K_steps):

            _, unc_c, _ = evidential_prob_unc(logits)

            fg = logits[:, 1:2]

            # neighbor logits
            nb = self.stencil(fg)

            B, N, C, D, H, W = nb.shape
            nb_flat = nb.view(B * N, C, D, H, W)

            nb_logits = torch.cat(
                [torch.zeros_like(nb_flat), nb_flat],
                dim=1,
            )
            _, unc_nb_flat, _ = evidential_prob_unc(nb_logits)
            unc_nb = unc_nb_flat.view(B, N, 1, D, H, W)

            fg_c = fg.unsqueeze(1)
            unc_c_b = unc_c.unsqueeze(1)

            # uncertainty gates
            gate_in = torch.sigmoid((unc_c_b - unc_nb) / (self.tau + 1e-8))
            gate_out = torch.sigmoid((unc_nb - unc_c_b) / (self.tau + 1e-8))

            # edge modulation
            feat_c = feat.unsqueeze(1)
            feat_nb = self.stencil_feat(feat)
            phi = self.edge_mlp(feat_c, feat_nb)
            phi = torch.tanh(phi)

            # flux
            flow_in = (gate_in * phi * nb).sum(dim=1)
            flow_out = (gate_out * fg_c).sum(dim=1)

            src_gate = torch.sigmoid((self.u0 - unc_c) / (self.tau + 1e-8))

            if self.use_source:
                fg = (
                    fg
                    + self.eta * (flow_in - flow_out)
                    + self.eta * src_gate * (logits0[:, 1:2] - fg)
                )
            else:
                fg = fg + self.eta * (flow_in - flow_out)

            logits = torch.cat([logits[:, 0:1], fg], dim=1)

        return logits


# =========================================================
# Fixed 6-neighborhood stencil
# =========================================================

class DepthwiseStencil3D(nn.Module):
    """Fixed 6-neighborhood depthwise stencil."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

        self.conv = nn.Conv3d(
            channels,
            channels * 6,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )

        self._init_weights()

        for p in self.parameters():
            p.requires_grad_(False)

    def _init_weights(self):
        w = torch.zeros((self.channels * 6, 1, 3, 3, 3))
        for c in range(self.channels):
            b = c * 6
            w[b + 0, 0, 1, 1, 2] = 1
            w[b + 1, 0, 1, 1, 0] = 1
            w[b + 2, 0, 1, 2, 1] = 1
            w[b + 3, 0, 1, 0, 1] = 1
            w[b + 4, 0, 2, 1, 1] = 1
            w[b + 5, 0, 0, 1, 1] = 1
        self.conv.weight.data.copy_(w)

    def forward(self, x):
        y = self.conv(x)
        B, _, D, H, W = y.shape
        return y.view(B, 6, self.channels, D, H, W)


# =========================================================
# Edge MLP
# =========================================================

class EdgeMLP(nn.Module):
    """Edge-wise modulation in decision-aligned feature space."""

    def __init__(self, feat_channels):
        super().__init__()
        self.linear = nn.Linear(feat_channels, 1, bias=False)
        nn.init.normal_(self.linear.weight, 0.0, 0.01)

    def forward(self, feat_c, feat_nb):
        diff = feat_c - feat_nb
        B, N, C, D, H, W = diff.shape

        diff = diff.permute(0,1,3,4,5,2).contiguous().view(-1, C)
        phi = self.linear(diff)
        phi = phi.view(B, N, D, H, W, 1).permute(0,1,5,2,3,4)

        return phi


# =========================================================
# Evidential uncertainty
# =========================================================

def evidential_prob_unc(logits):
    """
    Compute evidential probability and uncertainty.

    Returns:
        prob_fg: foreground probability
        unc: epistemic uncertainty
        S: Dirichlet strength
    """
    K = logits.shape[1]

    evidence = F.softplus(logits)
    alpha = evidence + 1.0
    S = alpha.sum(dim=1, keepdim=True)

    prob_fg = alpha[:, 1:2] / (S + 1e-8)
    unc = K / (S + 1e-8)

    return prob_fg, unc, S