#!/usr/bin/env python3

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchinfo

from profiti.PROFITI import ProFITi
from profiti.utils import profiti_collate_fn


def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("training.log")],
    )
    return logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Training Script for ProFITi",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training parameters
    training_group = parser.add_argument_group("Training Parameters")
    training_group.add_argument(
        "--epochs", "-e", type=int, default=500, help="Maximum number of epochs"
    )
    training_group.add_argument(
        "--batch-size", "-bs", type=int, default=128, help="Training batch size"
    )
    training_group.add_argument(
        "--learn-rate", "-lr", type=float, default=1e-3, help="Learning rate"
    )
    training_group.add_argument(
        "--betas",
        "-b",
        nargs=2,
        type=float,
        default=[0.9, 0.999],
        help="Adam optimizer betas",
    )
    training_group.add_argument(
        "--weight-decay",
        "-wd",
        type=float,
        default=1e-3,
        help="Weight decay for regularization",
    )
    training_group.add_argument(
        "--seed", "-s", type=int, default=42, help="Random seed for reproducibility"
    )

    # Model parameters
    model_group = parser.add_argument_group("Model Parameters")
    model_group.add_argument(
        "--hidden-size", "-hs", type=int, default=32, help="Hidden layer size"
    )
    model_group.add_argument(
        "--nlayers", "-nl", type=int, default=4, help="Number of layers"
    )
    model_group.add_argument(
        "--attn-head", "-ahd", type=int, default=1, help="Number of attention heads"
    )
    model_group.add_argument(
        "--latent-dim", "-ldim", type=int, default=64, help="Latent dimension size"
    )
    model_group.add_argument(
        "--flayers", "-fl", type=int, default=3, help="Number of flow layers"
    )

    # Dataset parameters
    data_group = parser.add_argument_group("Dataset Parameters")
    data_group.add_argument(
        "--dataset",
        "-dset",
        type=str,
        default="physionet2012",
        choices=["physionet2012", "mimiciii", "mimiciv"],
        help="Dataset to use",
    )
    data_group.add_argument(
        "--forc-time", "-ft", type=int, default=0, help="Forecast horizon in hours"
    )
    data_group.add_argument(
        "--cond-time", "-ct", type=int, default=36, help="Conditioning range in hours"
    )
    data_group.add_argument(
        "--nfolds",
        "-nf",
        type=int,
        default=5,
        help="Number of folds for cross-validation",
    )
    data_group.add_argument(
        "--fold", "-f", type=int, default=0, help="Current fold number"
    )

    # Additional parameters
    misc_group = parser.add_argument_group("Miscellaneous")
    misc_group.add_argument(
        "--output-dir",
        type=str,
        default="saved_models",
        help="Directory to save models",
    )
    misc_group.add_argument(
        "--early-stop-patience", type=int, default=30, help="Early stopping patience"
    )
    misc_group.add_argument(
        "--scheduler-patience", type=int, default=10, help="Scheduler patience"
    )

    return parser.parse_args()


def setup_environment(seed: int) -> torch.device:
    """Setup environment and random seeds."""
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # CUDA optimizations
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Enable anomaly detection for debugging
    torch.autograd.set_detect_anomaly(True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def load_dataset(args: argparse.Namespace):
    """Load the specified dataset."""
    dataset_config = {
        "normalize_time": True,
        "condition_time": args.cond_time,
        "forecast_horizon": args.forc_time,
        "num_folds": args.nfolds,
    }

    if args.dataset == "mimiciii":
        from tsdm.tasks.mimic_iii_debrouwer2019 import MIMIC_III_DeBrouwer2019

        return MIMIC_III_DeBrouwer2019(**dataset_config)
    elif args.dataset == "mimiciv":
        from tsdm.tasks.mimic_iv_bilos2021 import MIMIC_IV_Bilos2021

        return MIMIC_IV_Bilos2021(**dataset_config)
    elif args.dataset == "physionet2012":
        from tsdm.tasks.physionet2012 import Physionet2012

        return Physionet2012(**dataset_config)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


def create_dataloaders(task, fold: int, batch_size: int):
    """Create train, validation, and test dataloaders."""
    base_config = {
        "pin_memory": True,
        "num_workers": 4,
        "collate_fn": profiti_collate_fn,
    }

    train_loader = task.get_dataloader(
        (fold, "train"),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **base_config,
    )

    eval_config = {
        **base_config,
        "batch_size": 128,
        "shuffle": False,
        "drop_last": False,
    }
    valid_loader = task.get_dataloader((fold, "valid"), **eval_config)
    test_loader = task.get_dataloader((fold, "test"), **eval_config)

    return train_loader, valid_loader, test_loader


class Trainer:
    """Elegant trainer class for ProFITi model."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        logger: logging.Logger,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logger
        self.best_val_loss = float("inf")
        self.early_stop_counter = 0

    def train_epoch(self, train_loader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for batch in train_loader:
            # Move batch to device
            batch = [tensor.to(self.device) for tensor in batch]
            TX, CX, X, MX, TQ, CQ, Y, MQ = batch

            # Forward pass
            self.optimizer.zero_grad()
            self.model.distribution(TX, CX, X, MX, TQ, CQ, MQ)
            njNLL = self.model.compute_njNLL(Y, MQ)

            # Backward pass
            loss = njNLL.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Accumulate statistics
            total_loss += njNLL.sum().item()
            total_samples += TX.shape[0]

        return total_loss / total_samples

    def evaluate(self, data_loader) -> float:
        """Evaluate model on given data loader."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in data_loader:
                batch = [tensor.to(self.device) for tensor in batch]
                TX, CX, X, MX, TQ, CQ, Y, MQ = batch

                self.model.distribution(TX, CX, X, MX, TQ, CQ, MQ)
                njNLL = self.model.compute_njNLL(Y, MQ)

                total_loss += njNLL.sum().item()
                total_samples += TX.shape[0]

        return total_loss / total_samples

    def save_checkpoint(self, filepath: str, epoch: int, args: argparse.Namespace):
        """Save model checkpoint."""
        torch.save(
            {
                "args": args,
                "epoch": epoch,
                "state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": self.best_val_loss,
            },
            filepath,
        )

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        return checkpoint


def main():
    """Main training function."""
    # Setup
    args = parse_arguments()
    logger = setup_logging()
    device = setup_environment(args.seed)

    logger.info(f"Arguments: {args}")
    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load dataset and create dataloaders
    task = load_dataset(args)
    train_loader, valid_loader, test_loader = create_dataloaders(
        task, args.fold, args.batch_size
    )

    # Initialize model
    model_config = {
        "input_dim": task.dataset.shape[-1],
        "attn_head": args.attn_head,
        "latent_dim": args.latent_dim,
        "n_layers": args.nlayers,
        "f_layers": args.flayers,
        "device": device,
    }

    model = ProFITi(**model_config).to(device)
    logger.info(f"Model summary:\n{torchinfo.summary(model)}")

    # Initialize optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.learn_rate,
        betas=args.betas,
        weight_decay=args.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=args.scheduler_patience,
        factor=0.5,
        min_lr=1e-5,
        verbose=True,
    )

    # Initialize trainer
    trainer = Trainer(model, optimizer, scheduler, device, logger)

    # Generate experiment ID and model path
    experiment_id = int(time.time() * 1000) % 10000000
    model_path = output_dir / f"{args.dataset}_fold{args.fold}_{experiment_id}.pt"

    logger.info(f"Starting training with experiment ID: {experiment_id}")

    # Training loop
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # Train
        train_loss = trainer.train_epoch(train_loader)

        # Validate
        val_loss = trainer.evaluate(valid_loader)

        epoch_time = time.time() - start_time

        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Time: {epoch_time:.2f}s"
        )

        # Save best model and check early stopping
        if val_loss < trainer.best_val_loss:
            trainer.best_val_loss = val_loss
            trainer.save_checkpoint(str(model_path), epoch, args)
            trainer.early_stop_counter = 0
            logger.info(f"New best model saved with val_loss: {val_loss:.6f}")
        else:
            trainer.early_stop_counter += 1

        # Early stopping
        if trainer.early_stop_counter >= args.early_stop_patience:
            logger.info(
                f"Early stopping after {args.early_stop_patience} epochs without improvement"
            )
            break

        # Update scheduler
        scheduler.step(val_loss)

    # Final evaluation on test set
    logger.info("Loading best model for final evaluation...")
    trainer.load_checkpoint(str(model_path))

    start_time = time.time()
    test_loss = trainer.evaluate(test_loader)
    eval_time = time.time() - start_time

    logger.info(f"Final Results:")
    logger.info(f"Best Val Loss: {trainer.best_val_loss:.6f}")
    logger.info(f"Test Loss: {test_loss:.6f}")
    logger.info(f"Evaluation Time: {eval_time:.2f}s")

    return {
        "best_val_loss": trainer.best_val_loss,
        "test_loss": test_loss,
        "experiment_id": experiment_id,
    }


if __name__ == "__main__":
    main()
