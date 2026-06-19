"""
UGCP network based on a 3D U-Net backbone.

This implementation is intended for binary segmentation and keeps the
uncertainty-guided conservative propagation stage near logit space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import UNet


class UnetUGCP(nn.Module):
    """
    3D U-Net with UGCP refinement for binary segmentation.

    - The backbone outputs intermediate feature maps.
    - Two 1x1 heads project backbone features to:
      1. binary logits
      2. decision-aligned 2-channel features
    - UGCP refinement updates the foreground logit iteratively.
    """

    def __init__(
            self,
            in_channels=1,
            backbone_feat_out_ch=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="INSTANCE",
            act="RELU",
    ):
        super().__init__()

        ugcp_steps = 2
        ugcp_eta = 1.0
        ugcp_u0 = 0.5
        ugcp_tau = 0.1
        self.use_ugcp = True
        self.ugcp_use_source_term = True

        self.backbone = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=backbone_feat_out_ch,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            norm=norm,
            act=act,
        )

        self.logit_head = nn.Conv3d(
            backbone_feat_out_ch,
            2,
            kernel_size=1,
            bias=True,
        )
        self.feat_head = nn.Conv3d(
            backbone_feat_out_ch,
            2,
            kernel_size=1,
            bias=True,
        )

        if self.use_ugcp:
            self.ugcp = UQFluxRefine(
                K_steps=ugcp_steps,
                eta=ugcp_eta,
                u0=ugcp_u0,
                tau=ugcp_tau,
                source_item=self.ugcp_use_source_term,
                feat_head_channel=2,
            )

    def forward(self, x):
        backbone_features = self.backbone(x)  # [B, C, D, H, W]
        aligned_feat = self.feat_head(backbone_features)  # [B, 2, D, H, W]
        logits = self.logit_head(backbone_features)  # [B, 2, D, H, W]

        if self.use_ugcp:
            refined_logits = self.ugcp(logits, aligned_feat)
        else:
            refined_logits = logits

        return refined_logits


class UQFluxRefine(nn.Module):
    """
    Uncertainty-gated conservative refinement on binary logits.

    The module iteratively updates the foreground logit using:
    - fixed 6-neighborhood aggregation
    - uncertainty-based directional gates
    - feature-based edge modulation
    """

    def __init__(
            self,
            K_steps=2,
            eta=1.0,
            u0=0.5,
            tau=0.1,
            source_item=True,
            feat_head_channel=2,
    ):
        super().__init__()
        self.K_steps = K_steps
        self.eta = eta
        self.u0 = u0
        self.tau = tau
        self.ugcp_use_source_term = source_item

        self.stencil = DepthwiseStencil3D(1)  # foreground logit only
        self.stencil_f = DepthwiseStencil3D(feat_head_channel)  # aligned feature
        self.edge_mlp = EdgeMLP(feat_head_channel)

    def forward(self, logits, aligned_feat):
        """
        Refine binary logits with uncertainty-guided propagation.
        """
        refined_logits = logits

        for _ in range(self.K_steps):
            _, center_uncertainty, _ = evidential_prob_unc(refined_logits)
            foreground_logit = refined_logits[:, 1:2]

            neighbor_foreground_logits = self.stencil(foreground_logit)
            batch_size, num_neighbors, channels, depth, height, width = (
                neighbor_foreground_logits.shape
            )
            neighbor_fg_flat = neighbor_foreground_logits.view(
                batch_size * num_neighbors, channels, depth, height, width
            )

            neighbor_logits = torch.cat(
                [torch.zeros_like(neighbor_fg_flat), neighbor_fg_flat],
                dim=1,
            )
            _, neighbor_uncertainty_flat, _ = evidential_prob_unc(neighbor_logits)
            neighbor_uncertainty = neighbor_uncertainty_flat.view(
                batch_size, num_neighbors, 1, depth, height, width
            )

            center_logit = foreground_logit.unsqueeze(1)
            center_uncertainty_expanded = center_uncertainty.unsqueeze(1)

            gate_j2i = torch.sigmoid(
                (center_uncertainty_expanded - neighbor_uncertainty) / (self.tau + 1e-8)
            )
            gate_i2j = torch.sigmoid(
                (neighbor_uncertainty - center_uncertainty_expanded) / (self.tau + 1e-8)
            )

            center_feat = aligned_feat.unsqueeze(1)
            neighbor_feat = self.stencil_f(aligned_feat)
            edge_weight = self.edge_mlp(center_feat, neighbor_feat)
            edge_weight = torch.tanh(edge_weight)

            flow_in = (gate_j2i * edge_weight * neighbor_foreground_logits).sum(dim=1)
            flow_out = (gate_i2j * center_logit).sum(dim=1)

            source_gate = torch.sigmoid((self.u0 - center_uncertainty) / (self.tau + 1e-8))

            if self.ugcp_use_source_term:
                foreground_logit = (
                        foreground_logit
                        + self.eta * (flow_in - flow_out)
                        + self.eta * source_gate * (logits[:, 1:2] - foreground_logit)
                )
            else:
                foreground_logit = foreground_logit + self.eta * (flow_in - flow_out)

            refined_logits = torch.cat([refined_logits[:, 0:1], foreground_logit], dim=1)

        return refined_logits


class DepthwiseStencil3D(nn.Module):
    """
    Fixed 3D depthwise stencil that extracts the 6-neighborhood.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

        self.conv = nn.Conv3d(
            in_channels=channels,
            out_channels=channels * 6,
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
            base = c * 6
            w[base + 0, 0, 1, 1, 2] = 1.0
            w[base + 1, 0, 1, 1, 0] = 1.0
            w[base + 2, 0, 1, 2, 1] = 1.0
            w[base + 3, 0, 1, 0, 1] = 1.0
            w[base + 4, 0, 2, 1, 1] = 1.0
            w[base + 5, 0, 0, 1, 1] = 1.0
        self.conv.weight.data.copy_(w)

    def forward(self, x):
        y = self.conv(x)
        batch_size, _, depth, height, width = y.shape
        return y.view(batch_size, 6, self.channels, depth, height, width)


class EdgeMLP(nn.Module):
    """
    Edge-wise modulation in decision-aligned feature space.
    """

    def __init__(self, feat_head_channel):
        super().__init__()
        self.linear = nn.Linear(feat_head_channel, 1, bias=False)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)

    def forward(self, center_feat, neighbor_feat):
        feat_diff = center_feat - neighbor_feat
        batch_size, num_neighbors, channels, depth, height, width = feat_diff.shape
        feat_diff = feat_diff.permute(0, 1, 3, 4, 5, 2).contiguous().view(-1, channels)
        edge_weight = self.linear(feat_diff)
        edge_weight = edge_weight.view(
            batch_size, num_neighbors, depth, height, width, 1
        ).permute(0, 1, 5, 2, 3, 4)
        return edge_weight


def evidential_prob_unc(score):
    """
    Compute foreground probability and evidential uncertainty for binary logits.
    """
    num_classes = score.shape[1]
    evidence = F.softplus(score)
    alpha = evidence + 1.0
    evidence_sum = alpha.sum(dim=1, keepdim=True)
    prob_fg = alpha[:, 1:2] / (evidence_sum + 1e-8)
    unc = num_classes / (evidence_sum + 1e-8)
    return prob_fg, unc, evidence_sum
