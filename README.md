# ProFITi

This is the source code for the paper ''ProFITi: Probabilistic Forecasting of Irregular Time series with Conditional Flows''

# Requirements
python		3.8.11

Pytorch		1.9.0

sklearn		0.0

numpy		1.19.3


# Training and Evaluation

We provide an example using ''phsyionet``.

```
python train_profiti.py --epochs 500 --learn-rate 0.001 --batch-size 64 --attn-head 2 --latent-dim 32 --nlayers 3 --dataset physionet2012 --flayers 10
```

To download MIMIC-III and MIMIC-IV, a permission is required. Once, the datasets are downloaded, add them to the folder .tsdm/rawdata/ and use the TSDM package to extract the folds. TSDM is originally provided by Scholz .et .al from [https://openreview.net/forum?id=a-bD9-0ycs0] and modified by Yalavarthi .et. al (https://github.com/yalavarthivk/GraFITi/).
