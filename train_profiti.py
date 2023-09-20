import numpy as np
import torch
import sys
import os
import argparse
import time
from random import SystemRandom
import random
import torchinfo

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
print(' '.join(sys.argv))
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

if ARGS.dataset=="mimiciii":
    from tsdm.tasks.mimic_iii_debrouwer2019 import MIMIC_III_DeBrouwer2019
    TASK = MIMIC_III_DeBrouwer2019(normalize_time=True, condition_time=ARGS.cond_time, forecast_horizon = ARGS.forc_time, num_folds=ARGS.nfolds)
elif ARGS.dataset=="mimiciv":
    from tsdm.tasks.mimic_iv_bilos2021 import MIMIC_IV_Bilos2021
    TASK = MIMIC_IV_Bilos2021(normalize_time=True, condition_time=ARGS.cond_time, forecast_horizon = ARGS.forc_time, num_folds=ARGS.nfolds)
elif ARGS.dataset=='physionet2012':
    from tsdm.tasks.physionet2012 import Physionet2012
    TASK = Physionet2012(normalize_time=True, condition_time=ARGS.cond_time, forecast_horizon = ARGS.forc_time, num_folds=ARGS.nfolds)


from profiti.utils import profiti_collate_fn

dloader_config_train = {
    "batch_size": ARGS.batch_size,
    "shuffle": True,
    "drop_last": True,
    "pin_memory": True,
    "num_workers": 4,
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

from profiti.utils import compute_losses, compute_marginal_losses
losses = compute_losses(ARGS.std)
marginal_losses = compute_marginal_losses(ARGS.std)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from profiti.profiti import ProFITi

MODEL_CONFIG = {
    "input_dim": TASK.dataset.shape[-1],
    "attn_head": ARGS.attn_head,
    "latent_dim": ARGS.latent_dim,
    "n_layers": ARGS.nlayers,
    "f_layers": ARGS.flayers,
    "Lambda": ARGS.Lambda,
    "use_prelu": ARGS.use_prelu,
    "use_alpha": ARGS.use_alpha,
    "use_lrelu": ARGS.use_lrelu,
    "device": DEVICE
}

MODEL = ProFITi(**MODEL_CONFIG).to(DEVICE)
torchinfo.summary(MODEL)

def predict_fn(model, batch, marginal = 0):
    """Get targets and predictions."""
    T, X, M, TY, Y, MY = (tensor.to(DEVICE) for tensor in batch)
    # pdb.set_trace()
    Z, Jdet = model(T, X, M, TY, Y, MY, marginal)
    return Y, MY, Z, Jdet


# Reset
MODEL.zero_grad(set_to_none=True)

# ## Initialize Optimizer
from torch.optim import AdamW

OPTIMIZER = AdamW(MODEL.parameters(), **OPTIMIZER_CONFIG)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(OPTIMIZER, 'min', patience=10, factor=0.5, min_lr=0.00001, verbose=True)
es = False
best_val_loss = 10e8
early_stop = 0
for epoch in range(1, ARGS.epochs+1):
    loss_list = []
    count, train_nll = 0,0
    start_time = time.time()
    # training
    MODEL.train()
    for batch in (TRAIN_LOADER):
        # print('batch')
        OPTIMIZER.zero_grad()
        Y, MY, Z, Jdet = predict_fn(MODEL, batch)
        # pdb.set_trace()
        NLL = losses.loss(Y, MY, Z, Jdet)
        mask_sum = MY.sum((1,2)).bool().sum()
        train_nll += NLL*mask_sum
        count += mask_sum
        # Backward
        NLL.backward()
        OPTIMIZER.step()
    epoch_time = time.time()
    print('epoch: {}, train_nll: {:.4f}, epoch_time: {:.2f}'.format(
        epoch,
        train_nll/count,
        epoch_time - start_time
    ))
    count = 0
    val_nll = 0

    MODEL.eval()
    with torch.no_grad():
        # validation
        for batch in (VALID_LOADER):
            Y, MY, Z, Jdet = predict_fn(MODEL, batch)
            NLL = losses.loss(Y, MY, Z, Jdet)
            mask_sum = MY.sum((1,2)).bool().sum()
            val_nll += NLL*mask_sum
            count += mask_sum
    print('val_nll: {: .6f}'.format(
        val_nll/count,
    ))
    val_loss = val_nll/count
    # saving best model
    # break
    if val_loss < best_val_loss:
        best_val_loss = val_loss
    
        torch.save({    'args': ARGS,
                        'epoch': epoch,
                        'state_dict': MODEL.state_dict(),
                        'optimizer_state_dict': OPTIMIZER.state_dict(),
                        'loss': val_loss,
                    }, 'saved_models/'+ARGS.dataset + '_' + str(experiment_id) + '.h5')
        early_stop = 0
    else:
        early_stop += 1
    if early_stop == 30:
        print("Early stopping because of no improvement in val. metric for 30 epochs")
        es = True
    scheduler.step(val_loss)

    # LOGGER.log_epoch_end(epoch)
    if (epoch == ARGS.epochs) or (es == True):
        chp = torch.load('saved_models/' + ARGS.dataset + '_' + str(experiment_id) + '.h5')
        MODEL.load_state_dict(chp['state_dict'])
        count = 0
        test_nll = 0
        with torch.no_grad():
            # testing on the best model
            eval_time_start = time.time()
            for batch in (TEST_LOADER):
                # Forward
                Y, MY, Z, Jdet = predict_fn(MODEL, batch)
                NLL = losses.loss(Y, MY, Z, Jdet)
                mask_sum = MASK.sum((1,2)).bool().sum()
                test_nll += NLL*mask_sum
                count += mask_sum
              
        eval_time_end = time.time()
        sample_eval_time = eval_time_end - eval_time_start
        print('test_nll: {: .6f}, eval_time: {: .2f}s'.format(
        test_nll/count,
        sample_eval_time))
        print("Best_val_loss: ",best_val_loss.item(), " test_loss : ", (test_nll/count).item())
        break