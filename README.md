# ProFITi

This is the source code for the paper ''[ProFITi: Probabilistic Forecasting of Irregular Time series with Conditional Flows](https://arxiv.org/abs/2402.06293)''

# Requirements
python	3.10.16

torch	2.2.0

numpy	1.26.4

scipy	1.15.2

pandas	1.5.0

matplotlib	3.8.2

Please see `profiti_conda.yaml` file for more information

# Training and Evaluation

We provide an example using ''phsyionet``.

Following script can be used to train for normalized joint negative log-likelihood

```
python train_profiti.py --epochs 500 --learn-rate 0.001 --batch-size 64 --attn-head 2 --latent-dim 32 --nlayers 3 --dataset physionet2012 --flayers 10
```

Following script can be used to train for marginal negative log-likelihood

```
python train_profiti.py --epochs 500 --learn-rate 0.001 --batch-size 64 --attn-head 2 --latent-dim 32 --nlayers 3 --dataset physionet2012 --flayers 10 -mt
```

Onece the model is trained, following script can be used for additional evaluations such as MSE, CRPS or Energy Score the model (train-seed can be found in the training logs)

For joints:
```
python additional_evaluations.py --epochs 500 --learn-rate 0.001 --batch-size 64 --attn-head 2 --latent-dim 32 --nlayers 3 --dataset physionet2012 --flayers 10 --train-seed 8982595
```

For marginals:
```
python additional_evaluations.py --epochs 500 --learn-rate 0.001 --batch-size 64 --attn-head 2 --latent-dim 32 --nlayers 3 --dataset physionet2012 --flayers 10 -mt --train-seed 7896586
```

Note that, when model is trained for joints, it gives njnll whereas when it is trained for marginals, it gives mnll

To download MIMIC-III and MIMIC-IV, a permission is required. Once, the datasets are downloaded, add them to the folder .tsdm/rawdata/ and use the TSDM package to extract the folds. TSDM is originally provided by Scholz .et .al from [https://openreview.net/forum?id=a-bD9-0ycs0] and modified by Yalavarthi .et. al (https://github.com/yalavarthivk/GraFITi/).
