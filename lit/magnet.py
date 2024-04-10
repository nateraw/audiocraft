from .musicgen import LitMusicGen

import torch.nn.functional as F
from einops import rearrange
import torch
import numpy as np
import random
import math


class LitMagnet(LitMusicGen):
    def __init__(self, config):
        super().__init__(config)

    def setup(self, stage):
        self.hparams.transformer_lm.segment_duration = self.hparams.dataset.segment_duration
        self.hparams.transformer_lm.span_len = self.hparams.masking.span_len
        # sequence_len = int(self.hparams.dataset.segment_duration * self.compression_model.frame_rate)
        sequence_len = int(self.hparams.dataset.segment_duration * 150)  # hard coded for 48k encodec
        self.mean_maskrate_to_u = torch.tensor(self._calc_mean_maskrate_to_u_LUT(sequence_len), device=self.device)
        super().setup(stage)

    def _calc_mean_maskrate_to_u_LUT(self, T: int):
        """ Create a Look Up Table (LUT) transforming a discrete masking percentage m in 0,1,...,100 to u,
            the number of overlapping spans of length L to place s.t. the masking rate is approximately m/float(100).
            It first creates the inverse transformation, of the masking rate as function of u,
            using the expression choose(T - L, u) / choose(T, u), where L is the atomic span length used
            during masking. See https://arxiv.org/abs/2401.04577,
            appendix C, for the mean mask rate derivation.

            We leverage the fact that:
                                choose(T - L, u) / choose(T, u) = Prod_{j = 0}^{u - 1}((T - L - j)/(T - j))
            in the provided implementation, in order to avoid overflow.
        Args:
            T (float): Sequence length.
        Returns:
            (List) A LUT transforming m in 0,1,...,100 to u,
            s.t. the masking rate of the span-L mask is approximately m/float(100).
        """

        L = self.hparams.masking.span_len

        u2mean = [0.0]  # mean mask rate is 0.0 for u = 0
        v = (T - L) / float(T)
        for u in range(1, T):
            u2mean.append(1 - v)
            v *= (T - L - u) / (T - u)  # Overflow-safe implementation of choose(T - L, u) / choose(T, u).

        mean2u = []
        for maskperc in range(101):
            maskrate = maskperc / float(100)
            u = int(np.searchsorted(u2mean, maskrate))
            mean2u.append(u)

        return mean2u

    def _non_spans_mask(self, mask_probs: torch.Tensor, B: int, T: int, device: torch.device) -> torch.Tensor:
        """ Construct a boolean mask of shape [B, T, 1], with masking rates defined by mask_probs.
            The masked tokens are singletons, placed uniformly at random.
        Args:
            mask_probs (torch.Tensor): The desired masking rate per sample, of shape [B,]
            B (int): Batch size.
            T (int): Sequence length.
            device (torch.device): device of the output tensor
        Returns:
            (torch.Tensor): A mask of shape [B, T]
        """
        num_token_masked = (T * mask_probs).round().clamp(min=1)
        batch_randperm = torch.rand((B, T), device=device).argsort(dim=-1)
        return batch_randperm < rearrange(num_token_masked, 'b -> b 1')

    def _spans_mask(self, mask_probs: torch.Tensor, B: int, T: int, device: torch.device) -> torch.Tensor:
        """ Construct a spans mask with masking rates defined by mask_probs,
            where the atomic span length ( > 1 ) is defined by cfg.masking.span_len.
        Args:
            mask_probs (torch.Tensor): The desired masking rate per sample, of shape [B,]
            B (int): Batch size.
            T (int): Sequence length.
            device (torch.device): device of the output tensor
        Returns:
            (torch.Tensor): A spans mask of shape [B, T]
        """
        rounded_probs = torch.round(100 * mask_probs).long()
        k = self.mean_maskrate_to_u[rounded_probs].clamp(min=1)  # k is the number of span starts

        # sample random span starts
        batch_randperm = torch.rand((B, T), device=device).argsort(dim=-1)
        mask = batch_randperm < rearrange(k, 'b -> b 1')
        B, T = mask.shape
        shifted_mask = mask.clone()
        for _ in range(self.hparams.masking.span_len - 1):
            shifted_mask = torch.concat((torch.full((B, 1), False, device=device), shifted_mask[:, :-1]), dim=1)
            mask = torch.logical_or(mask, shifted_mask)

        return mask

    def _get_mask(self, mask_probs: torch.Tensor, B: int, T: int, device: torch.device) -> torch.Tensor:
        """ Construct a boolean mask with masking rates defined by mask_probs, and atomic
            span length defined by cfg.masking.span_len.
        Args:
            mask_probs (torch.Tensor): The desired masking rate per sample, of shape [B,]
            B (int): Batch size.
            T (int): Sequence length.
            device (torch.device): device of the output tensor
        Returns:
            (torch.Tensor): A boolean tensor of shape [B, T]
        """
        if self.hparams.masking.span_len <= 1:
            return self._non_spans_mask(mask_probs, B, T, device)

        return self._spans_mask(mask_probs, B, T, device)

    def _compute_cross_entropy_magnet(self, logits: torch.Tensor,
                                      targets: torch.Tensor, mask: torch.Tensor, stage: torch.Tensor) -> torch.Tensor:
        """ Compute cross entropy between multi-codebook targets and model's logits.
        The cross entropy is computed only on a specific codebook, defined by the stage argument.
        Valid timesteps for each codebook are pulled from the mask, where invalid
        timesteps are set to 0.

        Args:
            logits (torch.Tensor): Model's logits of shape [B, K, T, card].
            targets (torch.Tensor): Target codes, of shape [B, K, T].
            mask (torch.Tensor): Mask for valid target codes, of shape [B, K, T].
            stage (torch.Tensor): The codebook (idx) that is being optimized, as a scalar tensor.
        Returns:
            ce (torch.Tensor): Cross entropy of the codebook that is being optimized.
        """
        assert logits.shape[:-1] == targets.shape
        assert mask.shape == targets.shape
        ce = torch.zeros([], device=targets.device)
        logits_k = logits[:, stage, ...].contiguous().view(-1, logits.size(-1))  # [B x T, card]
        targets_k = targets[:, stage, ...].contiguous().view(-1)  # [B x T]
        mask_k = mask[:, stage, ...].contiguous().view(-1)  # [B x T]

        IGNORE_IDX = -1
        targets_k[~mask_k] = IGNORE_IDX
        q_ce = F.cross_entropy(logits_k, targets_k, ignore_index=IGNORE_IDX)

        ce += q_ce
        return ce

    def training_step(self, batch, batch_idx):
        condition_tensors, audio_tokens, padding_mask = self._prepare_tokens_and_attributes(batch)

        B, K, T = audio_tokens.shape
        device = self.device

        # Choose the stage (codebook idx) for update, uniformly at random.
        stage_ = random.randint(0, K - 1)
        stage = torch.full((1, ), stage_, device=device)

        # masking
        rand_time = torch.zeros((B,), device=device).float().uniform_(0, 1)
        rand_mask_probs = torch.cos(rand_time * math.pi * 0.5)

        # stage mask
        stage_mask = self._get_mask(rand_mask_probs, B, T, device)  # [B, T]
        stage_mask = stage_mask.unsqueeze(1)  # [B, 1, T]

        # Keep all preceding codebooks.
        mask = torch.full((B, K, T), False, device=device)
        mask[:, stage, :] = stage_mask

        # Mask all codebooks larger than stage_
        mask_id = self.model.special_token_id
        mask[:, (stage_+1):, :] = torch.full((B, K - stage_ - 1, T), True, device=device)
        input_tokens = torch.where(mask, mask_id, audio_tokens)

        # Take loss only on the chosen stage, and only on the masked tokens.
        loss_mask = torch.full((B, K, T), False, device=device)
        loss_mask[:, stage, :] = stage_mask

        model_output = self.model.compute_predictions(input_tokens, [], condition_tensors, stage=stage_)
        logits = model_output.logits
        loss_mask &= padding_mask
        ce = self._compute_cross_entropy_magnet(logits, audio_tokens, loss_mask, stage)

        self.log_dict(
            {
                "ce": ce,
                "ppl": torch.exp(ce),
                "lr": self.lr_schedulers().get_last_lr()[0]
            },
            prog_bar=True
        )
        return ce
