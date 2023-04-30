<h3 align="center">
<p> MINDG: A Drug-Target Interaction Prediction Method Based on an Integrated Learning Algorithm </h3>

This repository contains script which were used to build and train the MINDG model together with the scripts for evaluating the model's performance.

### Get Started

------

#### Dependency

```
numpy                   1.19.1
tqdm 					4.64.1
torch                   1.5.0
torch_sparse            0.6.4
pandas                  1.3.5
scikit-learn            0.22.1
matplotlib              3.2.2
scipy                   1.5.0
texttable               1.6.2
requests				2.28.2
pandas-flavor			0.5.0
subword-nmt				0.3.8
scipy  					1.7.3
prettytable          	3.6.0
texttable 				1.6.7
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

#### HDN options

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

