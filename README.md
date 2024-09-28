# scalable charge density prediction [NeurIPS 2024]

`scdp` is a codebase for training charge density prediction models described in the [paper](https://openreview.net/forum?id=b7REKaNUTv):

```
@inproceedings{
fu2024recipe,
title={A Recipe for Charge Density Prediction},
author={Fu, Xiang and Rosen, Andrew and Bystrom, Kyle and Wang, Rui and Musaelian, Albert and Kozinsky, Boris and Smidt, Tess and Jaakkola, Tommi},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=b7REKaNUTv}
}
```

please contact [Xiang Fu](mailto:xiangfu@meta.com) or open an issue if you have any questions.

### install dependencies

```
pip install --user -r requirements.txt
```

then install `scdp`:

```
pip install -e ./
```

### data downloading and processing

download the QM9 charge density dataset to `DATAPATH` from: https://data.dtu.dk/articles/dataset/QM9_Charge_Densities_and_Energies_Calculated_with_VASP/16794500. `DATAPATH` should contain all the `.tar` files.

preproces data to Lmdbs (no virtual nodes):

```
python scdp/scripts/preprocess.py --disable_pbc --data_path DATAPATH --out_path LMDBPATH --tar --device cpu --atom_cutoff 6 --vnode_method none
```

preproces data to Lmdbs (with bond midpoint virtual nodes):

```
python scdp/scripts/preprocess.py --disable_pbc --data_path DATAPATH --out_path LMDBPATH --tar --device cpu --atom_cutoff 6 --vnode_method bond 
```

`data_path` should be the path to the downloaded tar files. `out_path` is where the lmdb will be stored.

### training

make a `.env` file at the project root directory with the PROJECT_ROOT point to the project root. An example is provided in this repository.

configure config files to specify paths to the dataset lmdbs, split file, and checkpoint save paths. Then run:

```
python scdp/scripts/train.py 
```

by default we train an eSCN model with `K=8, L=6, beta=2.0` for 500000 steps over 4 GPUs using DDP. To finetune the same model with trainable scaling factors, run:

```
CUDA_VISIBLE_DEVICES=4,5,6,7 python scdp/scripts/train.py \
train=finetune model.expo_trainable=True
```

### inference

```
python scdp/scripts/test.py \
--data_path LMDBPATH \
 --split_file DATAPATH/datasplits.json \
--tag test --max_n_graphs 10000 --max_n_probe 400000 \
--ckpt_path CHECKPOINT_PATH
```

### pretrained models

we provide two pretrained models, downloadable from [Zenodo](https://zenodo.org/records/13146215):

- the fastest QM9 model described in the paper: 4-layer, Lmax=3, beta=2.0, no virtual nodes, no scaling factor finetuning.
- the most accurate QM9 model described in the paper: 8-layer, Lmax=6, beta=1.3, with virtual nodes, with scaling factor finetuning.
