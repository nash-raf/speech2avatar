import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_interpolation(features, seq_len):
    if features.shape[1] == seq_len:
        return features

    features = features.transpose(1, 2)
    output_features = F.interpolate(features, size=seq_len, align_corners=True, mode="linear")
    return output_features.transpose(1, 2)


class MimiContinuousEncoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.hidden_size = 512
        if opt.audio_feat_dim != self.hidden_size:
            raise ValueError(f"Continuous Mimi path requires --audio_feat_dim={self.hidden_size}")

    def _interpolate_batch(self, latent_tokens, seq_len, token_len=None):
        if token_len is None:
            return linear_interpolation(latent_tokens, seq_len)

        output = []
        for sample, length in zip(latent_tokens, token_len.tolist()):
            valid_length = max(int(length), 1)
            sample = sample[:valid_length].unsqueeze(0)
            output.append(linear_interpolation(sample, seq_len).squeeze(0))
        return torch.stack(output, dim=0)

    def forward(self, tokens, seq_len, token_len=None):
        if tokens.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected Mimi latents with feature dim {self.hidden_size}, got {tokens.shape[-1]}"
            )
        return self._interpolate_batch(tokens.float(), seq_len=seq_len, token_len=token_len)

    @torch.no_grad()
    def inference(self, tokens, seq_len):
        return self.forward(tokens, seq_len=seq_len)
