import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler

from lightning.pytorch import seed_everything

from ddpm import DDPM
from dataset import MNISTDataModule


def train():
    seed_everything(42, workers=True)
    device = 'auto'
    log_dir = 'logs'
    experiment = 'experiment'
    epochs = 2
    val_check_interval = None
    logger = TensorBoardLogger(save_dir=log_dir, name=experiment)

    callbacks = [
        ModelCheckpoint(
            dirpath=None,
            filename="epoch-{epoch:04d}",
            monitor="loss/mse",
            verbose=True,
            save_last=True,
            save_top_k=3,
            mode="min",
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor()
    ]
    profiler = SimpleProfiler(filename="profiler_report")
    trainer = pl.Trainer(
        accelerator=device,
        strategy="auto",
        devices="auto",
        num_nodes=1,
        precision="32-true",
        logger=logger,
        callbacks=callbacks,
        fast_dev_run=False,
        max_epochs=epochs,
        min_epochs=None,
        max_steps=-1,
        min_steps=None,
        max_time=None,
        limit_train_batches=None,
        limit_val_batches=None,
        limit_test_batches=None,
        limit_predict_batches=None,
        overfit_batches=0.0,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=None,
        log_every_n_steps=50,
        enable_checkpointing=None,
        enable_progress_bar=True,
        enable_model_summary=True,
        accumulate_grad_batches=1,
        gradient_clip_val=None,
        gradient_clip_algorithm="norm",
        deterministic=None,
        benchmark=None,
        inference_mode=True,
        use_distributed_sampler=True,
        profiler=profiler,
        detect_anomaly=False,
        barebones=False,
        plugins=None,
        sync_batchnorm=False,
        reload_dataloaders_every_n_epochs=0,
        default_root_dir=None,
    )
    module = DDPM()
    datamodule = MNISTDataModule()
    trainer.fit(model=module, datamodule=datamodule)


if __name__ == "__main__":
    train()
