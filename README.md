# Surrogate- and Invariance-Boosted Contrastive Learning (SIB-CL)
This repository contains all source code used to generate the results in the article [Surrogate- and invariance-boosted contrastive learning for data-scarce applications in science](https://arxiv.org/abs/2110.08406).

- The folder `generate_datasets` contains all numerical programs used to generate the datasets, for both Photonic Crystals (PhC) and the Time-independent Schrodinger Equation (TISE)
- `main.py` is the main code used to train the neural networks (explained in detail below)
<!-- - `get_training_log.py` plots the training curves using the saved log dictionaries; another option is to set the `--log_to_tensorboard` flag and monitor the curves using tensorboard
- `plot_results.py` plots the results using the saved results dictionaries in the format shown in the article
 -->

## Dependencies
Please install the required Python packages:
`pip install -r requirements.txt`.

A python3 environment can be created prior to this:
`conda create -n sibcl python=3.8; conda activate sibcl`.

Assess to MATLAB is required to calculate the density-of-states (DOS) of PhCs.

## Dataset Generation
### Photonic Crystals (PhCs)

Relevant code stored in `generate_datasets/PhC/`. 
Periodic unit cells are defined using a level set of a Fourier sum; different unit cells can be generated using the `get_random()` method of the `FourierPhC` class defined in `fourier_phc.py`.

To generate the labeled PhC datasets, we first compute their band structures using MPB. This can be executed via:

- **Target dataset of random Fourier unit cells:** 
  ```sh
  python phc_gendata.py --h5filename="mf1-s1" --pol="tm" --nsam=5000 --maxF=1 --seed=1`
  ```
- **Source dataset of simple cylinders:**
  ```sh
  python phc_gencylin.py --h5filename="cylin" --pol="tm" --nsam=10000`
  ```

each program will create a dataset with the eigen-frequencies, group velocities, etc, stored in a `.h5` file (which can be accessed using the h5py package). We then calculate the DOS using the GRR method provided by the MATLAB code from [/boyuanliuoptics/DOS-calculation/](https://github.com/boyuanliuoptics/DOS-calculation/blob/master/DOS_GGR.m). 
To do so, we first parse the data to create the `.txt` files required as inputs to the program, compute the DOS using MATLAB and then add the DOS labels back to the original `.h5` files. These steps will be executed automatically by simply running the shell script `get_DOS.sh` and entering the h5 filename identifiers when prompted. Note that for this to run smoothly, python and MATLAB will first need to be added to `PATH` and the environment with the installed dependencies should be loaded.

### Time-independent Schrodinger Equation (TISE)
Relevant code stored in `generate_datasets/TISE/`. Example usage:

- To generate target dataset, e.g. in 3D, `python tise_gendata.py --h5filename="tise3d" --ndim 3 --nsam 5000`.
- To generate low resolution dataset, `python tise_gendata.py --h5filename='tise3d_lr' --ndim 3 --nsam 10000 --lowres --orires=32` (`--orires` defines the resolution of the input to the neural network).
- To generate QHO dataset, `python tise_genqho.py --h5filename='tise2d_qho' --ndim 2 --nsam 10000`.

## SIB-CL and baselines training
Training of the neural networks for all problems introduced in the article (i.e. PhC DOS prediction, PhC Band structure prediction, TISE ground state energy prediction using both low resolution or QHO data as surrogate) can all be executed using `main.py` by indicating the appropriate flags (see below). This code also allows training via the SIB-CL framework or any of the baselines, again with the use of the appropriate flag. This code also contains other prediction problems not presented in the article, such as predicting higher energy states of TISE, TISE wavefunctions and single band structure.

### Important flags: 
- `--path_to_h5`: indicate directory where h5 datasets are located. The h5 filenames defined in the dataset classes in `datasets_PhC_SE.py` should also be modified according to the names used during dataset generation. 

- `--predict`: defines prediction task. Options: `'DOS'`, `'bandstructures'`, `'eigval'`, `'oneband'`, `'eigvec'`.

- `--train`: specify if training via SIB-CL or baselines. Options: `'sibcl'`, `'tl'`, `'sl'`, `'ssl'` (`'ssl'` performs regular contrastive learning without surrogate dataset). For invariance-boosted baselines, e.g. TL-I or SL-I, specify `'tl'` or `'sl'` here and add the relevant invariances flags (see below).

- `--iden`: required; specify identifier for saving of models, training logs and results.

Invariances flags: `--translate_pbc` (set this flag to include rolling translations), `--pg_uniform` (set this flag to uniformly sample the point group symmetry transformations), `--scale` (set this flag to scale unit cell - used for PhC), `--rotate` (set this flag to do 4-fold rotations), `--flip` (set this flag to perform horizontal and vertical mirrors). If `--pg_uniform` is used, there is no need to include `--rotate` and `--flip`.

Other optional flags can be displayed via `python main.py --help`. 
Examples of shell scripts can be found in the `sh_scripts` folder.

### Training outputs:
By default, running `main.py` will create 3 subdirectories:
- `./pretrained_models/`: state dictionaries of pretrained models at various epochs indicated in the `eplist` variable will be saved to this directory. These models are used for further fine-tuning.
- `./dicts/`: stores the evaluation losses on the test set as dictionaries saved as `.json` files. The results can then be plotted using `plot_results.py`.
- `./tlogs/`: training curves for pre-training and fine-tuning are stored in dictionaries saved as `.json` files. The training curves can be plotted using `get_training_logs.py`. Alternatively, the `--log_to_tensorboard` flag can be set and training curves can be viewed using tensorboard; in this case, the dictionaries will not be generated.


