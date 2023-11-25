import argparse
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch import seed_everything
import torch

from ddpm import DDPM
from dataset import MNISTDataModule


def parse_args():
    parser = argparse.ArgumentParser(prog='DDPM trainer')
    parser.add_argument(
        '--device', type=str,
        default='auto',
        help='Execution device',
    )
    parser.add_argument(
        '--epochs', type=int,
        default=10,
        help='Epochs to train',
    )
    parser.add_argument(
        '--logdir', type=str,
        default='logs',
        help='Path to train logs',
    )
    parser.add_argument(
        '--batch_size', type=int,
        default=128,
        help='Train batch size',
    )
    parser.add_argument(
        '--checkpoint', type=str,
        default=None,
        help='Abs path to fine-tuning checkpoint',
    )
    parser.add_argument(
        '--lr', type=float,
        default=1e-4,
        help='Learning rate',
    )
    parser.add_argument(
        '--hint', type=str,
        default='experiment',
        help='Name of the training session',
    )
    return parser.parse_args()


def train(args):
    seed_everything(42, workers=True)
    logger = TensorBoardLogger(save_dir=args.logdir, name=args.hint)

    callbacks = [
        ModelCheckpoint(
            dirpath=None,
            filename='epoch-{epoch:04d}-step-{step}',
            monitor='loss/mse',
            verbose=True,
            save_last=True,
            save_top_k=3,
            mode='min',
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor()
    ]
    profiler = SimpleProfiler(filename='profiler_report')
    trainer = pl.Trainer(
        accelerator=args.device,
        strategy='auto',
        devices='auto',
        num_nodes=1,
        precision='32-true',
        logger=logger,
        callbacks=callbacks,
        fast_dev_run=False,
        max_epochs=args.epochs,
        min_epochs=None,
        max_steps=-1,
        min_steps=None,
        max_time=None,
        limit_train_batches=None,
        limit_val_batches=None,
        limit_test_batches=None,
        limit_predict_batches=None,
        overfit_batches=0.0,
        val_check_interval=None,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=None,
        log_every_n_steps=50,
        enable_checkpointing=None,
        enable_progress_bar=True,
        enable_model_summary=True,
        accumulate_grad_batches=1,
        gradient_clip_val=None,
        gradient_clip_algorithm='norm',
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
    model = DDPM(lr=args.lr)
    if args.checkpoint:
        print(f'Fine-tuning checkpoint is set {args.checkpoint}')
        model = DDPM.load_from_checkpoint(
            args.checkpoint,
            map_location=torch.device(args.device)
        )
    datamodule = MNISTDataModule(batch_size=args.batch_size)
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == '__main__':
    train(parse_args())
