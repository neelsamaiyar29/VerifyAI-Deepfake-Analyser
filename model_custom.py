from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18


class CrossModalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        resnet = resnet18(weights=None)
        self.video_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.video_fc = nn.Linear(512, 256)

        self.audio_encoder = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=4,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video = video.float()
        audio = audio.float()

        if video.ndim != 5:
            raise ValueError(
                f"Expected video with shape (B, T, 3, 224, 224), got {tuple(video.shape)}."
            )
        if audio.ndim != 3:
            raise ValueError(
                f"Expected audio with shape (B, 64, T), got {tuple(audio.shape)}."
            )

        batch_size, time_steps, channels, height, width = video.shape
        if channels != 3 or height != 224 or width != 224:
            raise ValueError(
                f"Expected video frames shaped (3, 224, 224), got {(channels, height, width)}."
            )
        if audio.shape[0] != batch_size or audio.shape[1] != 64:
            raise ValueError(
                f"Expected audio with leading shape ({batch_size}, 64, T), got {tuple(audio.shape)}."
            )
        if audio.shape[2] != time_steps:
            raise ValueError(
                "Video and audio sequence lengths must match before model forward: "
                f"video T={time_steps}, audio T={audio.shape[2]}."
            )

        video = video.reshape(batch_size * time_steps, channels, height, width)
        video_features = self.video_encoder(video)
        video_features = video_features.flatten(1)
        video_features = video_features.reshape(batch_size, time_steps, 512)
        video_features = self.video_fc(video_features)

        audio_features = self.audio_encoder(audio)
        audio_features = audio_features.permute(0, 2, 1)

        attn_out, _ = self.attn(audio_features, video_features, video_features)

        mean_features = torch.mean(attn_out, dim=1)
        variance = torch.var(attn_out, dim=1, unbiased=False) + 1e-6
        fused_features = torch.cat((mean_features, variance), dim=1)

        logits = self.classifier(fused_features)
        return logits, variance
