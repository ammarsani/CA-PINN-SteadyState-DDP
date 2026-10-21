# CA-PINN for steady-state ion-coupled drift-diffusion-Poisson modeling

Code and data for the paper:

> A. S. Ibrahim, N. Malekjani, C. Anitescu, P. Chaipunya, P. Siripongwutikorn, N. Thongprong,
> *A Conservation-Aware Physics-Informed Neural Network for Steady-State Ion-Coupled Drift-Diffusion-Poisson Modeling of Perovskite Devices*,
> manuscript submitted for publication, 2026.

The code solves the one-dimensional steady-state drift-diffusion-Poisson equations for a perovskite device with mobile ions using a conservation-aware physics-informed neural network (CA-PINN), and validates it against a finite-difference (FD) solver. The model and method are described in the paper.

## Requirements

Python 3 with the packages in `requirements.txt` (TensorFlow 2.20.0):

```bash
pip install -r requirements.txt
```

The results in the paper were computed on Google Colab with an NVIDIA T4 GPU.

## Usage

```bash
git clone https://github.com/ammarsani/CA-PINN-SteadyState-DDP.git
cd CA-PINN-SteadyState-DDP
jupyter lab
```

On Google Colab, run `!git clone https://github.com/ammarsani/CA-PINN-SteadyState-DDP.git` and `%cd CA-PINN-SteadyState-DDP` first. Run the notebooks from the repository folder.

The saved results are included in `results/`, so the numbered notebooks reproduce the tables and figures of the paper without retraining. To retrain, set the switches at the top of a notebook to `True` (`RUN_PREBIAS` and `RUN_FD`, or `RUN_PINN_CASES` and `RUN_FD_REFERENCE` in notebooks 02-04). The ablation notebooks 06a-06d retrain by default. One CA-PINN training run takes about 25 minutes on a T4 GPU. Because GPU computations are not exactly deterministic, retrained errors can differ slightly from the published values.

## Contents

| Notebook | Result in the paper |
|---|---|
| `CA_PINN_Solver.ipynb`, `FD_Solver.ipynb` | CA-PINN and FD solvers (called by the notebooks below) |
| `01_FD_Grid_Independence.ipynb` | FD grid refinement (Section 2.3) |
| `02_Architecture_Study.ipynb` | Tables 3-4 |
| `03_Activation_Study.ipynb` | Table 5 |
| `04_Adam_Epoch_Study.ipynb` | Table 6 |
| `05_Validation.ipynb` | Table 7, Figs. 3-4 |
| `06a`-`06d_Ablation_*.ipynb`, `06e_Ablation_Plot.ipynb` | Table 8, Fig. 5 |
| `07_P0_Sweep.ipynb` | Table 9, Fig. 6 |
| `08_Vi_Sweep.ipynb` | Table 10, Fig. 7 |
| `09_Band_Diagrams.ipynb` | Fig. 8 |

`validation_utils.py` computes the relative L2 errors. Each run folder in `results/` contains the PINN and FD profiles, the error metrics and the training loss history.

## Citation

Please cite the paper above if you use this code (see `CITATION.cff`).

## License

MIT License, see `LICENSE`.
