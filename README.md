<h3 align="center">
<p> MINDG: A Drug-Target Interaction Prediction Method Based on an Integrated Learning Algorithm </h3>

This repository contains script which were used to build and train the MINDG model together with the scripts for evaluating the model's performance.

### Get Started

------

#### Dependency

```python
numpy
pandas
tqdm
scikit-learn
wget
torch
torch_sparse            
matplotlib
requests
pandas-flavor
subword-nmt
scipy  
lifelines
prettytable
tensorboard
texttable 
```

#### Dataset

The dataset used in the experiments are provided in `/data`

#### HOAGCN options

```python
  --seed              INT     Random seed.                   Default is 42.
  --epochs            INT     Number of training epochs.     Default is 50.
  --batch_size        INT     Number of samples in a batch   Default is 256.
  --early-stopping    INT     Early stopping rounds.         Default is 10.
  --learning-rate     FLOAT   Adam learning rate.            Default is 5e-4.
  --dropout           FLOAT   Dropout rate value.            Default is 0.1.
  --order             INT     Order of neighbor including 0  Default is 3.
  --cuda              BOOL    Run on GPR                     Default is True.
```

#### MDN options

```python
cls_hidden_dims = [1024,1024,512], 
train_epoch = 100, 
LR = 0.001, 
batch_size = 128,
cnn_target_filters = [32,64,96],
cnn_target_kernels = [4,8,12]
```

#### Acknowledgement

The code is based on [HOGCN](https://github.com/kckishan/HOGCN-LP) and [DeepPurpose](https://github.com/kexinhuang12345/DeepPurpose).

