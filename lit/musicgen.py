from pathlib import Path
import time
import typing as tp
import warnings
from multiprocessing import cpu_count
from typing import Optional, Union

import math
import torch
from torch.nn import functional as F

from audiocraft.solvers import base, builders
from audiocraft.solvers.compression import CompressionSolver
from audiocraft import metrics as eval_metrics
from audiocraft import models
from audiocraft.data.audio_dataset import AudioDataset
from audiocraft.data.music_dataset import MusicDataset, MusicInfo, AudioInfo
from audiocraft.data.audio_utils import normalize_audio
from audiocraft.modules.conditioners import JointEmbedCondition, SegmentWithAttributes, WavCondition
from audiocraft.utils.cache import CachedBatchWriter, CachedBatchLoader
from audiocraft.utils.samples.manager import SampleManager
from audiocraft.utils.utils import get_dataset_from_loader, is_jsonable, warn_once, model_hash
from audiocraft.optim import fsdp

from encodec import EncodecModel
import pytorch_lightning as pl


class LitMusicGen(pl.LightningModule):
    DATASET_TYPE: builders.DatasetType = builders.DatasetType.MUSIC

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.model = None

        self._cached_batch_writer = None
        self._cached_batch_loader = None

    def configure_optimizers(self):
        optimizer = builders.get_optimizer(builders.get_optim_parameter_groups(self.model), self.hparams.optim)
        lr_scheduler = builders.get_lr_scheduler(optimizer, self.hparams.schedule, self.trainer.estimated_stepping_batches)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def setup(self, stage=None):
        if stage == "fit":
            if self.hparams.datasource.max_sample_rate == 48000:
                assert self.hparams.continue_from is None, "48khz model not supported with fine-tuning from facebook's pretrained checkpoints."
                self.compression_model = EncodecModel.encodec_model_48khz().to(self.device)
                self.compression_model.set_target_bandwidth(6.0)
            else:
                self.compression_model = CompressionSolver.wrapped_model_from_checkpoint(
                    self.hparams,
                    self.hparams.compression_model_checkpoint,
                    device=self.device
                )
            
            if self.hparams.continue_from is None:
                self.model = models.builders.get_lm_model(self.hparams).to(self.device)
            else:
                if self.hparams.solver == "musicgen":
                    print(f"Loading pretrained musicgen model: {self.hparams.continue_from}")
                    self.model = models.loaders.load_lm_model(self.hparams.continue_from, "cuda").to(self.device)
                else:
                    print(f"Loading pretrained magnet model: {self.hparams.continue_from}")
                    # 50 is frame rate...hard coded to 50, beware.
                    # Frame rate is `sample_rate // np.prod(encoder_ratios)`
                    # ex for musicgen: 
                    # 32000 // np.prod([8, 5, 4, 4]) == 50
                    # self.model = load_lm_model_magnet(self.hparams.continue_from, 50, "cuda").to(self.device)
                    self.model = models.loaders.load_lm_model_magnet(self.hparams.continue_from, 50, "cuda").to(self.device)
                    self.model.train()

            # HACK - Double check all this is necessary...
            self.model.device = self.device
            self.model.condition_provider.device = self.device
            self.model.condition_provider.to(self.device)
            for name in self.model.condition_provider.conditioners.keys():
                self.model.condition_provider.conditioners[name].device = self.device
                self.model.condition_provider.conditioners[name].to(self.device)

            if self.hparams.cache.path:
                if self.hparams.cache.write:
                    self._cached_batch_writer = CachedBatchWriter(Path(self.hparams.cache.path))
                    if self.hparams.cache.write_num_shards:
                        if self.global_rank == 0:
                            print("Multiple shard cache, best_metric_name will be set to None.")
                        self._best_metric_name = None
                else:
                    self._cached_batch_loader = CachedBatchLoader(
                        Path(self.hparams.cache.path), self.hparams.dataset.batch_size, self.hparams.dataset.num_workers,
                        min_length=self.hparams.optim.updates_per_epoch or 1)
                    self.dataloaders['original_train'] = self.dataloaders['train']
                    self.dataloaders['train'] = self._cached_batch_loader  # type: ignore
        self.dataloaders = builders.get_audio_datasets(self.hparams, dataset_type=self.DATASET_TYPE)

    def train_dataloader(self):
        return self.dataloaders["train"]

    def _prepare_tokens_and_attributes(
        self, batch: tp.Tuple[torch.Tensor, tp.List[SegmentWithAttributes]],
        check_synchronization_points: bool = False
    ) -> tp.Tuple[dict, torch.Tensor, torch.Tensor]:
        """Prepare input batchs for language model training.

        Args:
            batch (tuple[torch.Tensor, list[SegmentWithAttributes]]): Input batch with audio tensor of shape [B, C, T]
                and corresponding metadata as SegmentWithAttributes (with B items).
            check_synchronization_points (bool): Whether to check for synchronization points slowing down training.
        Returns:
            Condition tensors (dict[str, any]): Preprocessed condition attributes.
            Tokens (torch.Tensor): Audio tokens from compression model, of shape [B, K, T_s],
                with B the batch size, K the number of codebooks, T_s the token timesteps.
            Padding mask (torch.Tensor): Mask with valid positions in the tokens tensor, of shape [B, K, T_s].
        """
        if self.model.training:
            warnings.warn(
                "Up to version 1.0.1, the _prepare_tokens_and_attributes was evaluated with `torch.no_grad()`. "
                "This is inconsistent with how model were trained in the MusicGen paper. We removed the "
                "`torch.no_grad()` in version 1.1.0. Small changes to the final performance are expected. "
                "Really sorry about that.")
        if self._cached_batch_loader is None or self.current_stage != "train":
            audio, infos = batch
            audio = audio.to(self.device)
            audio_tokens = None
            assert audio.size(0) == len(infos), (
                f"Mismatch between number of items in audio batch ({audio.size(0)})",
                f" and in metadata ({len(infos)})"
            )
        else:
            audio = None
            # In that case the batch will be a tuple coming from the _cached_batch_writer bit below.
            infos, = batch  # type: ignore
            assert all([isinstance(info, AudioInfo) for info in infos])
            assert all([info.audio_tokens is not None for info in infos])  # type: ignore
            audio_tokens = torch.stack([info.audio_tokens for info in infos]).to(self.device)  # type: ignore
            audio_tokens = audio_tokens.long()
            for info in infos:
                if isinstance(info, MusicInfo):
                    # Careful here, if you want to use this condition_wav (e.b. chroma conditioning),
                    # then you must be using the chroma cache! otherwise the code will try
                    # to use this segment and fail (by that I mean you will see NaN everywhere).
                    info.self_wav = WavCondition(
                        torch.full([1, info.channels, info.total_frames], float('NaN')),
                        length=torch.tensor([info.n_frames]),
                        sample_rate=[info.sample_rate],
                        path=[info.meta.path],
                        seek_time=[info.seek_time])
                    dataset = get_dataset_from_loader(self.dataloaders['original_train'])
                    assert isinstance(dataset, MusicDataset), type(dataset)
                    if dataset.paraphraser is not None and info.description is not None:
                        # Hackingly reapplying paraphraser when using cache.
                        info.description = dataset.paraphraser.sample_paraphrase(
                            info.meta.path, info.description)
        # prepare attributes
        attributes = [info.to_condition_attributes() for info in infos]
        attributes = self.model.cfg_dropout(attributes)
        attributes = self.model.att_dropout(attributes)
        tokenized = self.model.condition_provider.tokenize(attributes)

        # Now we should be synchronization free.
        if self.device == "cuda" and check_synchronization_points:
            torch.cuda.set_sync_debug_mode("warn")

        if audio_tokens is None:
            with torch.no_grad():
                if self.hparams.sample_rate != 48000:
                    audio_tokens, scale = self.compression_model.encode(audio)
                    assert scale is None, "Scaled compression model not supported with LM."
                else:
                    encoded_frames = self.compression_model.encode(audio)
                    audio_tokens = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)

        # with self.autocast:
        condition_tensors = self.model.condition_provider(tokenized)

        # create a padding mask to hold valid vs invalid positions
        padding_mask = torch.ones_like(audio_tokens, dtype=torch.bool, device=audio_tokens.device)
        # replace encodec tokens from padded audio with special_token_id
        if self.hparams.tokens.padding_with_special_token:
            audio_tokens = audio_tokens.clone()
            padding_mask = padding_mask.clone()
            token_sample_rate = self.compression_model.frame_rate
            B, K, T_s = audio_tokens.shape
            for i in range(B):
                n_samples = infos[i].n_frames
                audio_sample_rate = infos[i].sample_rate
                # take the last token generated from actual audio frames (non-padded audio)
                valid_tokens = math.floor(float(n_samples) / audio_sample_rate * token_sample_rate)
                audio_tokens[i, :, valid_tokens:] = self.model.special_token_id
                padding_mask[i, :, valid_tokens:] = 0

        if self.device == "cuda" and check_synchronization_points:
            torch.cuda.set_sync_debug_mode("default")

        if self._cached_batch_writer is not None:
            assert self._cached_batch_loader is None
            assert audio_tokens is not None
            for info, one_audio_tokens in zip(infos, audio_tokens):
                assert isinstance(info, AudioInfo)
                if isinstance(info, MusicInfo):
                    assert not info.joint_embed, "joint_embed and cache not supported yet."
                    info.self_wav = None
                assert one_audio_tokens.max() < 2**15, one_audio_tokens.max().item()
                info.audio_tokens = one_audio_tokens.short().cpu()
            self._cached_batch_writer.save(infos)

        return condition_tensors, audio_tokens, padding_mask

    def on_train_start(self):
        if self._cached_batch_writer is not None:
            self._cached_batch_writer.start_epoch(self.current_epoch)
        if self._cached_batch_loader is None:
            dataset = get_dataset_from_loader(self.dataloaders['train'])
            assert isinstance(dataset, AudioDataset)
            dataset.current_epoch = self.current_epoch
        else:
            self._cached_batch_loader.start_epoch(self.current_epoch)

    def _compute_cross_entropy(
        self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
    ) -> tp.Tuple[torch.Tensor, tp.List[torch.Tensor]]:
        """Compute cross entropy between multi-codebook targets and model's logits.
        The cross entropy is computed per codebook to provide codebook-level cross entropy.
        Valid timesteps for each of the codebook are pulled from the mask, where invalid
        timesteps are set to 0.

        Args:
            logits (torch.Tensor): Model's logits of shape [B, K, T, card].
            targets (torch.Tensor): Target codes, of shape [B, K, T].
            mask (torch.Tensor): Mask for valid target codes, of shape [B, K, T].
        Returns:
            ce (torch.Tensor): Cross entropy averaged over the codebooks
            ce_per_codebook (list of torch.Tensor): Cross entropy per codebook (detached).
        """
        B, K, T = targets.shape
        assert logits.shape[:-1] == targets.shape
        assert mask.shape == targets.shape
        ce = torch.zeros([], device=targets.device)
        ce_per_codebook: tp.List[torch.Tensor] = []
        for k in range(K):
            logits_k = logits[:, k, ...].contiguous().view(-1, logits.size(-1))  # [B x T, card]
            targets_k = targets[:, k, ...].contiguous().view(-1)  # [B x T]
            mask_k = mask[:, k, ...].contiguous().view(-1)  # [B x T]
            ce_targets = targets_k[mask_k]
            ce_logits = logits_k[mask_k]
            q_ce = F.cross_entropy(ce_logits, ce_targets)
            ce += q_ce
            ce_per_codebook.append(q_ce.detach())
        # average cross entropy across codebooks
        ce = ce / K
        return ce, ce_per_codebook

    def training_step(self, batch, batch_idx):
        condition_tensors, audio_tokens, padding_mask = self._prepare_tokens_and_attributes(batch)
        model_output = self.model.compute_predictions(audio_tokens, [], condition_tensors)  # type: ignore
        logits = model_output.logits
        mask = padding_mask & model_output.mask
        ce, ce_per_codebook = self._compute_cross_entropy(logits, audio_tokens, mask)

        # log separately from rest of metrics as we don't want all of them to be in progress bar
        self.log("ce", ce, prog_bar=True)

        metrics = {}
        metrics["ppl"] = torch.exp(ce)
        metrics["lr"] = self.lr_schedulers().get_last_lr()[0]
        for k, ce_q in enumerate(ce_per_codebook):
            metrics[f'ce_q{k + 1}'] = ce_q
            metrics[f'ppl_q{k + 1}'] = torch.exp(ce_q)
        self.log_dict(metrics)
        return ce

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        assert gradient_clip_algorithm in ('norm', None), gradient_clip_algorithm
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_val)
        self.log("grad_norm", grad_norm, on_step=True)

    # TODO lol!
    # def validation_step(self, batch, batch_idx):
    #     pass
    #
    # def val_dataloader(self):
    #     return self.dataloaders["evaluate"]
