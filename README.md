# CA-PINN for steady-state ion-coupled drift-diffusion-Poisson modeling

Code and data for the paper:

> A. S. Ibrahim, N. Malekjani, C. Anitescu, P. Chaipunya, P. Siripongwutikorn, N. Thongprong,
> *A Conservation-Aware Physics-Informed Neural Network for Steady-State Ion-Coupled Drift-Diffusion-Poisson Modeling of Perovskite Devices*,
> manuscript submitted for publication, 2026.

The code solves the one-dimensional steady-state drift-diffusion-Poisson equations for a perovskite device with mobile ions using a conservation-aware physics-informed neural network (CA-PINN), and validates it against a finite-difference (FD) solver.

## Model

The perovskite layer occupies $0 \le x \le b$. At a fixed pre-bias voltage $V_i$, the electrostatic potential $\psi$, the electron density $n$ and the mobile-ion density $I$ satisfy

$$\frac{d^2\psi}{dx^2} = -\frac{q}{\varepsilon}(I - n), \qquad \frac{dJ_n}{dx} = 0, \qquad \frac{dF_I}{dx} = 0,$$

$$J_n = qD_n\left(\frac{dn}{dx} - \frac{n}{V_t}\frac{d\psi}{dx}\right), \qquad F_I = -D_I\left(\frac{dI}{dx} + \frac{I}{V_t}\frac{d\psi}{dx}\right),$$

with $\psi(0) = 0$, $\psi(b) = V_{bi} + V_i$, fixed electron densities $n_a = N_c e^{-\phi_a/V_t}$ and $n_c = N_c e^{-\phi_c/V_t}$ at the contacts, and ion-blocking contacts, $F_I(0) = F_I(b) = 0$. Because the ions cannot leave the device, the steady state is fixed only once the total ion content is prescribed by the mean ion density $P_0$.

The CA-PINN combines

1. logarithmic densities, $u_n = \ln n$ and $u_I = \ln I$, which keep both densities positive;
2. trial solutions that satisfy the Dirichlet conditions for $\psi$ and $n$ exactly;
3. a zero-flux condition for the ions at both contacts;
4. a global ion-conservation constraint, $\frac{1}{b}\int_0^b I\,dx = P_0$.

The FD reference uses the Scharfetter-Gummel scheme with Gummel iteration. Details are given in the paper.

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

The saved results are included in `results/`, so the numbered notebooks reproduce the tables and figures of the paper without retraining. To retrain, set the switches at the top of a notebook to `True` (`RUN_PREBIAS` and `RUN_FD`, or `RUN_PINN_CASES` and `RUN_FD_REFERENCE` in notebooks 02-04). The ablation notebooks 06a-06d retrain by default. One CA-PINN training run takes about 25 minutes on a T4 GPU, and retraining all studies takes about 40 hours. Because GPU computations are not exactly deterministic, retrained errors can differ slightly from the published values.

## Contents

| Notebook | Content |
|---|---|
| `CA_PINN_Solver.ipynb` | CA-PINN solver, called by the notebooks below |
| `FD_Solver.ipynb` | Finite-difference reference solver |
| `01_FD_Grid_Independence.ipynb` | Grid refinement of the FD solver |
| `02_Architecture_Study.ipynb` | Network depth and width |
| `03_Activation_Study.ipynb` | Activation functions |
| `04_Adam_Epoch_Study.ipynb` | Number of Adam epochs |
| `05_Validation.ipynb` | Validation against the FD solution |
| `06a`-`06d_Ablation_*.ipynb` | Ablation of log densities and ion conservation |
| `06e_Ablation_Plot.ipynb` | Ablation solution profiles |
| `07_P0_Sweep.ipynb` | Effect of the mobile-ion density $P_0$ |
| `08_Vi_Sweep.ipynb` | Effect of the pre-bias voltage $V_i$ |
| `09_Band_Diagrams.ipynb` | Energy-band diagrams |

`validation_utils.py` computes the relative L2 errors. Each run folder in `results/` contains the PINN and FD profiles, the error metrics and the training loss history.

## Citation

Please cite the paper above if you use this code (see `CITATION.cff`).

## License

MIT License, see `LICENSE`.
