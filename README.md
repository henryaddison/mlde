# ML Downscaling Emulator

A machine learning emulator of a CPM based on a diffusion model.

This is the code for the paper Addison et al. (2024) "Machine learning emulation of precipitation from km-scale regional climate simulations using a diffusion model".

Diffusion model implementation forked from PyTorch implementation for the paper [Score-Based Generative Modeling through Stochastic Differential Equations](https://openreview.net/forum?id=PxTIG12RRHS) by [Yang Song](https://yang-song.github.io), [Jascha Sohl-Dickstein](http://www.sohldickstein.com/), [Diederik P. Kingma](http://dpkingma.com/), [Abhishek Kumar](http://users.umiacs.umd.edu/~abhishek/), [Stefano Ermon](https://cs.stanford.edu/~ermon/), and [Ben Poole](https://cs.stanford.edu/~poole/).

## Dependencies

1. Clone repo and cd into it
2. Create conda environment: `conda env create -f environment.lock.yml` (or add dependencies to your own: `conda env install -f environment.txt`)
3. Activate the conda environment (if not already done so)
4. Install ml_downscaling_emulator locally: `pip install -e .`
5. \[Optional\] Install U-Net code: `git clone --depth 1 https://github.com/henryaddison/Pytorch-UNet.git src/ml_downscaling_emulator/unet` - this is only necessary if you wish to use the deterministic comparison models.
6. Configure application behaviour with environment variables. See `.env.example` for variables that can be set.

Any datasets are assumed to be found in `${DERIVED_DATA}/moose/nc-datasets/{dataset_name}/`. In particular, the config key config.data.dataset_name is the name of the dataset to use to train the model.

## Diffusion Model Usage

### Data

Datasets for use with the emulator can be created using [[https://github.com/henryaddison/mlde-data]].
This repo contains further information about dataset specification.
The datasets used in the paper can be found on [Zenodo](https://doi.org/10.5281/zenodo.11504859).

**NB** the interface commonly takes just the name of a dataset name. It is expected to be found at `${DERIVED_DATA}/moose/nc-datasets/{dataset_name}/` (where DERIVED_DATA is a configurable environment variable).

### Smoke test

```sh
tests/smoke-test
```

Uses a simpler network to test the full training and sampling regime.
Recommended to run with a sample of the dataset.

### Training

Train models through `bin/main.py`, e.g. to train the model used in the paper use

```sh
python bin/main.py --config src/ml_downscaling_emulator/score_sde_pytorch/configs/subvpsde/ukcp_local_pr_12em_cncsnpp_continuous.py --workdir ${DERVIED_DATA}/path/to/models/paper-12em --mode train
```

```sh
main.py:
  --mode: <train>: Running mode: train
  --workdir: Working directory for storing data related to model such as model snapshots, tranforms or samples
  --config: Training configuration.
    (default: 'None')
```

* `mode` is "train". When set to "train", it starts the training of a new model, or resumes the training of an old model if its meta-checkpoints (for resuming running after pre-emption in a cloud environment) exist in `workdir/checkpoints-meta`.

* `workdir` is the path that stores all artifacts of one experiment, like checkpoints, transforms and samples. Recommended to be a subdirectory of ${DERIVED_DATA}.

* `config` is the path to the config file. Config files for emulators are provided in `src/configs/`. They are formatted according to [`ml_collections`](https://github.com/google/ml_collections) and heavily based on ncsnpp config files.

  **Naming conventions of config files**: the path of a config file is a combination of the following dimensions:
  * SDE: `subvpsde`
  * data source: `ukcp_local`
  * variable: `pr`
  * ensemble members: `12em` (all 12) or `1em` (single)
  * model: `cncsnpp`
  * continuous: train the model with continuously sampled time steps.

Functionalities can be configured through config files, or more conveniently, through the command-line support of the `ml_collections` package.


### Sampling

Once have trained a model create samples from it with `bin/predict.py`, e.g.

```sh
python bin/predict.py --checkpoint epoch_20 --dataset bham_60km-4x_12em_psl-sphum4th-temp4th-vort4th_eqvt_random-season --split test  --ensemble-member 01 --input-transform-dataset bham_60km-4x_12em_psl-sphum4th-temp4th-vort4th_eqvt_random-season --input-transform-key pixelmmsstan --num-samples 1 ${DERVIED_DATA}/path/to/models/paper-12em
```

This example command will:
* use the checkpoint of the model in `${DERVIED_DATA}/path/to/models/paper-12em/checkpoints/{checkpoint}.pth` and model config from training `${DERVIED_DATA}/path/to/models/paper-12em/config.yml`.
* store samples generated in `${DERVIED_DATA}/path/to/models/paper-12em/samples/{dataset}/{input_transform_data}-{input_transform_key}/{split}/{ensemble_member}/`. Sample files ar named like `predictions-{uuid}.nc`.
* generate samples conditioned on examples from ensemble member `01` in the `test` subset of the `bham_60km-4x_12em_psl-sphum4th-temp4th-vort4th_eqvt_random-season` dataset.
* transform the inputs based on the `bham_60km-4x_12em_psl-sphum4th-temp4th-vort4th_eqvt_random-season` dataset using the `pixelmmsstan` approach.
* generate 1 set of samples.
