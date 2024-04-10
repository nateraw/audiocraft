from multiprocessing import cpu_count

import pytorch_lightning as pl
from lit import config_musicgen, config_magnet
from lit.musicgen import LitMusicGen
from lit.magnet import LitMagnet



def main():
    MODEL = "magnet"  # "musicgen"
    SIZE = "medium"  # "small"
    CONTINUE_FROM = "facebook/magnet-medium-10secs"  # "facebook/magnet-small-10secs"

    # TODO - make sure this is working properly. Saving works, but haven't tried loading yet.
    DUMMY_CACHE_RUN = False
    CACHE_PATH = "./audiocraft-jamendo-cache"

    LitModel = LitMagnet if MODEL == "magnet" else LitMusicGen
    cfg = config_magnet.cfg if MODEL == "magnet" else config_musicgen.cfg

    pl.seed_everything(cfg.seed)
    cfg.fsdp.use = True

    cfg.sample_rate = 32000
    cfg.datasource.max_sample_rate = 32000
    cfg.datasource.max_channels = 1
    cfg.datasource.train = "egs/splice_v2"
    cfg.datasource.valid = "egs/splice_v2"
    cfg.datasource.evaluate = "egs/splice_v2"
    cfg.datasource.generate = "egs/splice_v2"
    # cfg.optim.optimizer = "adam"
    # cfg.optim.lr = 1e-5
    cfg.schedule.cosine.warmup = 75

    cfg.dataset.num_workers = cpu_count()
    cfg.dataset.batch_size = 14
    accumulate_grad_batches = 8

    cfg.dataset.segment_duration = 10
    cfg.channels = 1
    # cfg.channels = 2
    # cfg.transformer_lm.card = 1024

    if DUMMY_CACHE_RUN:
        print(f"Running dummy cache run with cache path: {CACHE_PATH}. Setting model to xsmall.")
        cfg.cache.path = CACHE_PATH
        cfg.cache.write = True
        cfg.fsdp.use = False
        SIZE = "xsmall"

    if SIZE == "xsmall":
        cfg.transformer_lm.dim = 64
        cfg.transformer_lm.num_heads = 2
        cfg.transformer_lm.num_layers = 2
    # Small - 300M Param. Otherwise its medium - 1.3B
    if SIZE == "small":
        cfg.transformer_lm.dim = 1024
        cfg.transformer_lm.num_heads = 16
        cfg.transformer_lm.num_layers = 24
    # Medium - 1.3B
    elif SIZE == "medium":
        cfg.transformer_lm.dim = 1536
        cfg.transformer_lm.num_heads = 24
        cfg.transformer_lm.num_layers = 48

    cfg.continue_from = CONTINUE_FROM

    cfg.wandb.project = "audiocraft-48k-stereo-lightning"
    cfg.wandb.name = f"splicev2-{MODEL}-{cfg.datasource.max_sample_rate // 1000}k-stereo-{SIZE}" + (
        f"-ft-{CONTINUE_FROM}" if CONTINUE_FROM else ""
    )
    cfg.logging.log_wandb = True

    if cfg.fsdp.use:
        from audiocraft.modules.transformer import StreamingTransformerLayer
        from audiocraft.modules.conditioners import ConditioningProvider
        policy = {StreamingTransformerLayer, ConditioningProvider}
        # Orig code uses sharding_strategy="SHARD_GRAD_OP", but its not working for me with lightning
        strategy = pl.strategies.FSDPStrategy(auto_wrap_policy=policy, cpu_offload=False)
    else:
        strategy = "ddp_find_unused_parameters_true"

    model = LitModel(cfg)
    trainer = pl.Trainer(
        devices=1,
        precision="16-mixed",
        max_steps=1000,  # Steps reported in paper @ batch size 192
        # strategy=strategy,
        accumulate_grad_batches=8,
        log_every_n_steps=10,
        gradient_clip_val=cfg.optim.max_norm,
        callbacks=[
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
            pl.callbacks.ModelCheckpoint(
                dirpath=None,
                save_top_k=4,
                monitor='ce',
                mode='min',
                save_last=True,
                every_n_train_steps=200,
            ),
        ],
        logger=pl.loggers.WandbLogger(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
        ) if cfg.logging.log_wandb else None,
    )
    trainer.fit(model)  #ckpt_path="/path/to/lightning_logs/version_X/checkpoints/epoch=X-step=XX.ckpt")
    return cfg, model, trainer


if __name__ == '__main__':
    main()

# If we want to update encodec to match audiocraft encodec, the following attrs will need to be added:
# compression_model.frame_rate
# compression_model.sample_rate
# compression_model.encode/decode
# compression_model.cardinality
# compression_model.num_codebooks
# 7.420