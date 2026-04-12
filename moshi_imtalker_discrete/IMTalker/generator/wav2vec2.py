import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_interpolation(features, seq_len):
    if features.shape[1] == seq_len:
        return features

    features = features.transpose(1, 2)
    output_features = F.interpolate(features, size=seq_len, align_corners=True, mode="linear")
    return output_features.transpose(1, 2)


class MoshiTokenEncoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.hidden_size = opt.audio_feat_dim
        self.embedding = nn.Embedding(2048, self.hidden_size)

    def _interpolate_batch(self, embedded_tokens, seq_len, token_len=None):
        if token_len is None:
            return linear_interpolation(embedded_tokens, seq_len)

        output = []
        for sample, length in zip(embedded_tokens, token_len.tolist()):
            valid_length = max(int(length), 1)
            sample = sample[:valid_length].unsqueeze(0)
            output.append(linear_interpolation(sample, seq_len).squeeze(0))
        return torch.stack(output, dim=0)

    def forward(self, tokens, seq_len, token_len=None):
        embedded_tokens = self.embedding(tokens.long())
        return self._interpolate_batch(embedded_tokens, seq_len=seq_len, token_len=token_len)

    @torch.no_grad()
    def inference(self, tokens, seq_len):
        return self.forward(tokens, seq_len=seq_len)
