# IWRCL

## Requirements

### Training

The following packages are required only for training:

```text
torch
pandas
tqdm
matplotlib
```

Install them with:

```bash
pip install torch pandas tqdm matplotlib
```

If you are using an NVIDIA GPU, install the appropriate CUDA-enabled PyTorch build for your system.

### Evaluation

The following additional packages are used for evaluation and visualization:

```text
seaborn
scipy
umap
```

Install them with:

```bash
pip install seaborn scipy umap-learn
```

## Training

Training is handled by `main.py`.

In most cases, you only need to specify the dataset name:

```bash
python main.py --data normal
```

For example:

```bash
python main.py --data exp
python main.py --data mix
python main.py --data copula
python main.py --data cortex
python main.py --data VerbAgg
```

Dataset-specific model architectures and training hyperparameters are loaded automatically from the corresponding configuration file.

A small number of options can also be customized through `main.py`, such as the training method, number of importance samples, GPU/device, and random seed.

Example:

```bash
python main.py --data exp --forward IWRCL --num_samples 100 --gpu 0 --seed 42
```

## Evaluation

Evaluation code is provided as Jupyter notebooks in the `eval/` directory.

Refer to the corresponding `.ipynb` file for each experiment:

```text
eval/
├── normal.ipynb
├── exp.ipynb
├── mix.ipynb
├── copula.ipynb
├── cortex.ipynb
└── VerbAgg.ipynb
```

The notebooks load the trained outputs from the `save/` directory and reproduce the evaluation metrics and visualizations used for each experiment.
