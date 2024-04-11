from multiprocessing import cpu_count
from pathlib import Path

import pytorch_lightning as pl
from lit import config_musicgen, config_magnet
from lit.musicgen import LitMusicGen
from lit.magnet import LitMagnet


def main(
    model_type="musicgen",  # "musicgen"
    model_size="small",  # "small"
    base_model_id="facebook/musicgen-stereo-small",
    resume_from_ckpt=None,
    # TODO - make sure this is working properly. Saving works, but haven't tried loading yet.
    use_cache=False,
    cache_path="./audiocraft-jamendo-cache",
    use_fsdp=False,
    sample_rate=32000,
    channels=2,
    batch_size=2,
    num_workers=None,
    accumulate_grad_batches=1,
    segment_duration=10,
    log_wandb=True,
    wandb_project="audiocraft-48k-stereo-lightning",
    warmup_steps=75,
    ds_train="egs/splice_v2",
    ds_valid="egs/splice_v2",
    ds_evaluate="egs/splice_v2",
    ds_generate="egs/splice_v2",
    num_devices=1,
    log_every_n_steps=10,
    save_every_n_steps=200,
    precision="16-mixed",
    max_steps=1000,
    optimizer="dadam",
    lr=1
):
    LitModel = LitMagnet if model_type == "magnet" else LitMusicGen
    cfg = config_magnet.cfg if model_type == "magnet" else config_musicgen.cfg

    pl.seed_everything(cfg.seed)
    cfg.fsdp.use = use_fsdp
    cfg.sample_rate = sample_rate
    cfg.datasource.max_sample_rate = sample_rate
    cfg.datasource.max_channels = channels
    cfg.datasource.train = ds_train
    cfg.datasource.valid = ds_valid
    cfg.datasource.evaluate = ds_evaluate
    cfg.datasource.generate = ds_generate

    cfg.optim.optimizer = optimizer
    cfg.optim.lr = lr
    cfg.schedule.cosine.warmup = warmup_steps

    cfg.dataset.num_workers = num_workers or cpu_count()
    cfg.dataset.batch_size = batch_size
    accumulate_grad_batches = accumulate_grad_batches

    cfg.dataset.segment_duration = segment_duration
    cfg.channels = channels

    if cfg.channels == 2 and cfg.sample_rate != 48000:
        cfg.interleave_stereo_codebooks.use = True

    if use_cache:
        print(f"Running dummy cache run with cache path: {cache_path}. Setting model to xsmall.")
        cfg.cache.path = cache_path
        cfg.cache.write = True
        cfg.fsdp.use = False
        model_size = "xsmall"

    if model_size == "xsmall":
        cfg.transformer_lm.dim = 64
        cfg.transformer_lm.num_heads = 2
        cfg.transformer_lm.num_layers = 2
    # Small - 300M Param. Otherwise its medium - 1.3B
    if model_size == "small":
        cfg.transformer_lm.dim = 1024
        cfg.transformer_lm.num_heads = 16
        cfg.transformer_lm.num_layers = 24
    # Medium - 1.3B
    elif model_size == "medium":
        cfg.transformer_lm.dim = 1536
        cfg.transformer_lm.num_heads = 24
        cfg.transformer_lm.num_layers = 48

    cfg.continue_from = base_model_id

    cfg.wandb.project = wandb_project
    cfg.wandb.name = f"{Path(ds_train).stem}-{model_type}-{cfg.datasource.max_sample_rate // 1000}k-stereo-{model_size}" + (
        f"-ft-{base_model_id}" if base_model_id else ""
    )
    cfg.logging.log_wandb = log_wandb

    if cfg.fsdp.use:
        from audiocraft.modules.transformer import StreamingTransformerLayer
        from audiocraft.modules.conditioners import ConditioningProvider
        policy = {StreamingTransformerLayer, ConditioningProvider}
        # Orig code uses sharding_strategy="SHARD_GRAD_OP", but its not working for me with lightning
        strategy = pl.strategies.FSDPStrategy(auto_wrap_policy=policy, cpu_offload=False)
    elif num_devices > 1:
        strategy = "ddp_find_unused_parameters_true"
    else:
        strategy = "auto"

    model = LitModel(cfg)
    trainer = pl.Trainer(
        devices=num_devices,
        precision=precision,
        max_steps=max_steps,  # Steps reported in paper @ batch size 192
        strategy=strategy,
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=log_every_n_steps,
        gradient_clip_val=cfg.optim.max_norm,
        callbacks=[
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
            pl.callbacks.ModelCheckpoint(
                dirpath=None,
                save_top_k=3,
                monitor='ce',
                mode='min',
                save_last=True,
                every_n_train_steps=save_every_n_steps,
            ),
        ],
        logger=pl.loggers.WandbLogger(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
        ) if cfg.logging.log_wandb else None,
    )
    trainer.fit(model, ckpt_path=resume_from_ckpt)
    return cfg, model, trainer


if __name__ == '__main__':
    import fire
    fire.Fire(main)
