import numpy as np
import torch
import sys
import os
import argparse
import time
from random import SystemRandom
import random
import torchinfo


from torch.optim import AdamW
import pdb

torch.autograd.set_detect_anomaly(True)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)
# fmt: off
parser = argparse.ArgumentParser(description="Training Script for ProFITi.")
parser.add_argument("-e",  "--epochs",       default=500,    type=int,   help="maximum epochs")
parser.add_argument("-f",  "--fold",         default=0,      type=int,   help="fold number")
parser.add_argument("-bs", "--batch-size",   default=128,     type=int,   help="batch-size")
parser.add_argument("-lr", "--learn-rate",   default=0.001,  type=float, help="learn-rate")
parser.add_argument("-b",  "--betas", default=(0.9, 0.999),  type=float, help="adam betas", nargs=2)
parser.add_argument("-wd", "--weight-decay", default=0.001,  type=float, help="weight-decay")
parser.add_argument("-hs", "--hidden-size",  default=32,    type=int,   help="hidden-size")
parser.add_argument("-s",  "--seed",         default=None,   type=int,   help="Set the random seed.")
parser.add_argument("-nl",  "--nlayers", default=4,   type=int,   help="")
parser.add_argument("-ahd",  "--attn-head", default=1,   type=int,   help="")
parser.add_argument("-ldim",  "--latent-dim", default=64,   type=int,   help="")
parser.add_argument("-dset", "--dataset", default="physionet2012", type=str, help="Name of the dataset")
parser.add_argument("-ft", "--forc-time", default=0, type=int, help="forecast horizon in hours")
parser.add_argument("-ct", "--cond-time", default=36, type=int, help="conditioning range in hours")
parser.add_argument("-nf", "--nfolds", default=5, type=int, help="#folds for crossvalidation")
parser.add_argument("-fl", "--flayers", default=3, type=int, help="number of layers in the flow")
parser.add_argument("--std", default=1, type=float, help='standard devaiation of base gaussian')
parser.add_argument("--marginal", default=1, type=int, help='do we want marginal?')
parser.add_argument("--use-prelu", default=0, type=int, help='use prelu')
parser.add_argument("--use-alpha", default=1, type=int, help='use alpha')
parser.add_argument("--use-lrelu", default=0, type=int, help='use lrelu')
# fmt: on
ARGS = parser.parse_args()
print(" ".join(sys.argv))
experiment_id = int(SystemRandom().random() * 10000000)
print(ARGS, experiment_id)

if ARGS.seed is not None:
    torch.manual_seed(ARGS.seed)
    random.seed(ARGS.seed)
    np.random.seed(ARGS.seed)

OPTIMIZER_CONFIG = {
    "lr": ARGS.learn_rate,
    "betas": torch.tensor(ARGS.betas),
    "weight_decay": ARGS.weight_decay,
}

if ARGS.dataset == "mimiciii":
    from tsdm.tasks.mimic_iii_debrouwer2019 import MIMIC_III_DeBrouwer2019

    TASK = MIMIC_III_DeBrouwer2019(
        normalize_time=True,
        condition_time=ARGS.cond_time,
        forecast_horizon=ARGS.forc_time,
        num_folds=ARGS.nfolds,
    )
elif ARGS.dataset == "mimiciv":
    from tsdm.tasks.mimic_iv_bilos2021 import MIMIC_IV_Bilos2021

    TASK = MIMIC_IV_Bilos2021(
        normalize_time=True,
        condition_time=ARGS.cond_time,
        forecast_horizon=ARGS.forc_time,
        num_folds=ARGS.nfolds,
    )
elif ARGS.dataset == "physionet2012":
    from tsdm.tasks.physionet2012 import Physionet2012

    TASK = Physionet2012(
        normalize_time=True,
        condition_time=ARGS.cond_time,
        forecast_horizon=ARGS.forc_time,
        num_folds=ARGS.nfolds,
    )


from profiti.utils import profiti_collate_fn

dloader_config_train = {
    "batch_size": ARGS.batch_size,
    "shuffle": True,
    "drop_last": True,
    "pin_memory": False,
    "num_workers": 0,
    "collate_fn": profiti_collate_fn,
}

dloader_config_infer = {
    "batch_size": 128,
    "shuffle": False,
    "drop_last": False,
    "pin_memory": True,
    "num_workers": 0,
    "collate_fn": profiti_collate_fn,
}

TRAIN_LOADER = TASK.get_dataloader((ARGS.fold, "train"), **dloader_config_train)
VALID_LOADER = TASK.get_dataloader((ARGS.fold, "valid"), **dloader_config_infer)
TEST_LOADER = TASK.get_dataloader((ARGS.fold, "test"), **dloader_config_infer)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from profiti.PROFITI import ProFITi

MODEL_CONFIG = {
    "input_dim": TASK.dataset.shape[-1],
    "attn_head": ARGS.attn_head,
    "latent_dim": ARGS.latent_dim,
    "n_layers": ARGS.nlayers,
    "f_layers": ARGS.flayers,
    "device": DEVICE,
}

MODEL = ProFITi(**MODEL_CONFIG).to(DEVICE)
torchinfo.summary(MODEL)

OPTIMIZER = AdamW(MODEL.parameters(), **OPTIMIZER_CONFIG)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    OPTIMIZER, "min", patience=10, factor=0.5, min_lr=0.00001, verbose=True
)
ES = False
best_val_loss = 10e8
early_stop = 0
for epoch in range(1, ARGS.epochs + 1):
    count, train_njNLL = 0, 0
    start_time = time.time()
    # training
    MODEL.train()
    for batch in TRAIN_LOADER:
        OPTIMIZER.zero_grad()
        TX, CX, X, MX, TQ, CQ, Y, MQ = (tensor.to(DEVICE) for tensor in batch)
        MODEL.distribution(TX, CX, X, MX, TQ, CQ, MQ)
        njNLL = MODEL.compute_njNLL(Y, MQ)
        count += TX.shape[0]
        train_njNLL += njNLL.sum()
        print(njNLL.mean().item(), end="\r")
        njNLL.mean().backward()
        # torch.nn.utils.clip_grad_norm_(MODEL.parameters(), max_norm=1.0)
        OPTIMIZER.step()

    epoch_time = time.time()
    print(
        "epoch: {}, train_njNLL: {:.4f}, epoch_time: {:.2f}".format(
            epoch, train_njNLL / count, epoch_time - start_time
        )
    )

    MODEL.eval()
    with torch.no_grad():
        # validation
        VALID_njNLL, COUNT = 0, 0
        for batch in VALID_LOADER:
            TX, CX, X, MX, TQ, CQ, Y, MQ = (tensor.to(DEVICE) for tensor in batch)
            MODEL.distribution(TX, CX, X, MX, TQ, CQ, MQ)
            njNLL = MODEL.compute_njNLL(Y, MQ)
            COUNT += TX.shape[0]
            VALID_njNLL += njNLL.sum()
    val_loss = VALID_njNLL / count
    print("val_nll: {: .6f}".format(val_loss))

    # saving best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(
            {
                "args": ARGS,
                "epoch": epoch,
                "state_dict": MODEL.state_dict(),
                "optimizer_state_dict": OPTIMIZER.state_dict(),
                "loss": val_loss,
            },
            "saved_models/" + ARGS.dataset + "_" + str(experiment_id) + ".h5",
        )
        early_stop = 0
    else:
        early_stop += 1
    if early_stop == 30:
        print("Early stopping because of no improvement in val. metric for 30 epochs")
        ES = True
    scheduler.step(val_loss)

    # LOGGER.log_epoch_end(epoch)
    if (epoch == ARGS.epochs) or (ES == True):
        chp = torch.load(
            "saved_models/" + ARGS.dataset + "_" + str(experiment_id) + ".h5"
        )
        MODEL.load_state_dict(chp["state_dict"])
        COUNT = 0
        TEST_njNLL = 0
        with torch.no_grad():
            # testing on the best model
            eval_time_start = time.time()
            for batch in TEST_LOADER:
                # Forward
                TX, CX, X, MX, TQ, CQ, Y, MQ = (tensor.to(DEVICE) for tensor in batch)
                MODEL.distribution(TX, CX, X, MX, TQ, CQ, MQ)
                njNLL = MODEL.compute_njNLL(Y, MQ)
                COUNT += TX.shape[0]
                TEST_njNLL += njNLL.sum()
        test_loss = TEST_njNLL / count

        eval_time_end = time.time()
        sample_eval_time = eval_time_end - eval_time_start
        print(
            "test_nll: {: .6f}, eval_time: {: .2f}s".format(test_loss, sample_eval_time)
        )
        print(f"Best_val_loss: {best_val_loss.item()}, test_loss: {test_loss.item()}")
