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

from profiti.model import ProFITi
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
        description="Evaluation Script for ProFITi",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training parameters
    training_group = parser.add_argument_group("Training Parameters")
    training_group.add_argument(
        "--epochs", "-e", type=int, default=500, help="Maximum number of epochs"
    )
    training_group.add_argument(
        "--batch-size", "-bs", type=int, default=32, help="Training batch size"
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
    model_group.add_argument(
        "--marginal-training",
        "-mt",
        action="store_true",
        default=False,
        help="Use marginal training objective",
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
    misc_group.add_argument(
        "--train-seed",
        type=int,
        default=10,
        help="Logging interval in epochs",
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


class Evaluator:
    """Evaluator class for ProFITi model."""

    def __init__(
        self,
        model: nn.Module,
        args,
        device: torch.device,
        logger: logging.Logger,
    ):
        self.model = model
        self.device = device
        self.logger = logger
        self.args = args

    def evaluate(self, data_loader) -> float:
        """Evaluate model on given data loader."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in data_loader:
                batch = [tensor.to(self.device) for tensor in batch]
                tx, cx, x, mx, tq, cq, y, mq = batch

                self.model.distribution(tx, cx, x, mx, tq, cq, mq)
                if self.args.marginal_training:
                    mnll = self.model.compute_mnll(y, mq)
                    current_queries = mq.sum()
                    total_loss += (
                        mnll.item()
                    )  # mnll is already summed for all the variables
                    total_samples += current_queries
                else:
                    njnll = self.model.compute_njnll(y, mq)
                    total_loss += njnll.sum().item()
                    total_samples += tx.shape[0]

        return total_loss / total_samples

    def evaluate_additional_metrics(
        self, data_loader, nsamples: int
    ) -> Tuple[float, float]:
        """Evaluate additional metrics (MSE, Robust MSE, CRPS, Energy Score) on given data loader.
        This function also collects all samples, targets, and masks for further analysis.
        """
        self.model.eval()
        total_mse = 0.0
        total_robust_mse = 0.0
        total_crps = 0.0
        total_energy_score = 0.0
        total_samples = 0
        total_queries = 0
        all_samples = []
        all_targets = []
        all_masks = []

        with torch.no_grad():
            for batch in data_loader:
                batch = [tensor.to(self.device) for tensor in batch]
                tx, cx, x, mx, tq, cq, y, mq = batch

                self.model.distribution(tx, cx, x, mx, tq, cq, mq)
                samples, _ = self.model.samples(mq, nsamples=nsamples)  # unused log_det
                mse = self.model.mse(y, mq, nsamples=nsamples)
                robust_mse = self.model.robust_mse(y, mq, nsamples=nsamples)
                crps = self.model.crps(y, mq, nsamples=nsamples)
                energy_score = self.model.energy_score(y, mq, nsamples=nsamples)

                current_queries = mq.sum().item()
                current_instances = tx.shape[0]
                total_mse += mse * current_queries
                total_robust_mse += robust_mse * current_queries
                total_crps += crps * current_queries
                total_energy_score += energy_score * current_instances
                total_queries += current_queries
                total_samples += current_instances

                all_samples.append(samples.cpu())
                all_masks.append(mq.cpu())
                all_targets.append(y.cpu())
            return (
                total_mse / total_queries,
                total_robust_mse / total_queries,
                total_crps / total_queries,
                total_energy_score / total_samples,
                all_samples,
                all_targets,
                all_masks,
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

    logger.info("Arguments: %s", args)
    logger.info("Using device: %s", device)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load dataset and create dataloaders
    task = load_dataset(args)
    _, _, test_loader = create_dataloaders(task, args.fold, args.batch_size)

    # Initialize model
    model_config = {
        "input_dim": task.dataset.shape[-1],
        "attn_head": args.attn_head,
        "latent_dim": args.latent_dim,
        "n_layers": args.nlayers,
        "f_layers": args.flayers,
        "marginal_training": args.marginal_training,
        "device": device,
    }

    model = ProFITi(**model_config).to(device)
    logger.info("Model summary:\n%s", torchinfo.summary(model))

    # Initialize trainer
    trainer = Evaluator(model, args, device, logger)

    # Generate experiment ID and model path
    experiment_id = args.train_seed
    model_path = output_dir / f"{args.dataset}_fold{args.fold}_{experiment_id}.pt"

    # Final evaluation on test set
    logger.info("Loading best model for final evaluation...")
    trainer.load_checkpoint(str(model_path))

    logger.info("computing test likelihood")
    start_time = time.time()
    test_loss = trainer.evaluate(test_loader)
    eval_time = time.time() - start_time

    # Compute additional metrics mse, robust mse, crps, energy score

    logger.info("computing additional metrics")
    test_mse, test_robust_mse, crps, energy_score, _, _, _ = (
        trainer.evaluate_additional_metrics(test_loader, nsamples=100)
    )

    # IF YOU WANT TO COMPUTE POINT METRICS ALONG WITH SAMPLES (UNCOMMENT BELOW)
    # test_mse, test_robust_mse, crps, energy_score, all_samples, all_targets, all_masks = (
    # trainer.evaluate_additional_metrics(test_loader)
    # )

    logger.info("Final Results:")
    logger.info("Evaluation Time: %.2fs for likelihood", eval_time)
    logger.info("Marginal results: %s", str(args.marginal_training))
    logger.info("Test likelihood loss: %.6f", test_loss)
    logger.info("Test MSE: %.6f", test_mse)
    logger.info("Test Robust MSE: %.6f", test_robust_mse)
    logger.info("Test CRPS: %.6f", crps)
    logger.info("Test Energy Score: %.6f", energy_score)


if __name__ == "__main__":
    print(" ".join(sys.argv))
    main()
