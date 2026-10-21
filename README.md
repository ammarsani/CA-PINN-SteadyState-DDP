# CA-PINN for steady-state ion-coupled drift-diffusion-Poisson modeling

Code and result data for the paper:

> A. S. Ibrahim, N. Malekjani, C. Anitescu, P. Chaipunya, P. Siripongwutikorn, N. Thongprong,
> *A Conservation-Aware Physics-Informed Neural Network for Steady-State Ion-Coupled Drift-Diffusion-Poisson Modeling of Perovskite Devices*,
> manuscript submitted for publication, 2026.

The code solves the one-dimensional steady-state drift-diffusion-Poisson (DDP) equations for an electron-only perovskite device with mobile ions (halide vacancies) using a conservation-aware physics-informed neural network (CA-PINN). The network predicts the electrostatic potential $\psi(x)$, the electron density $n(x)$ and the mobile-ion density $I(x)$. A finite-difference (FD) solver based on the Scharfetter-Gummel scheme with Gummel iteration is included as the reference solution.

The saved results are included in `results/`, so all tables and figures of the paper can be regenerated without retraining the networks.

## Model

The device occupies $0 \le x \le b$, with the anode at $x=0$ and the cathode at $x=b$. At a fixed pre-bias voltage $V_i$ the steady-state equations are

$$\frac{d^2\psi}{dx^2} = -\frac{q}{\varepsilon}(I - n), \qquad \frac{dJ_n}{dx} = 0, \qquad \frac{dF_I}{dx} = 0,$$

$$J_n = qD_n\left(\frac{dn}{dx} - \frac{n}{V_t}\frac{d\psi}{dx}\right), \qquad F_I = -D_I\left(\frac{dI}{dx} + \frac{I}{V_t}\frac{d\psi}{dx}\right),$$

with $\psi(0) = 0$ and $\psi(b) = V_{bi} + V_i$, electron densities $n_a = N_c e^{-\phi_a/V_t}$ and $n_c = N_c e^{-\phi_c/V_t}$ at the contacts, and blocking contacts for the ions, $F_I(0) = F_I(b) = 0$.

The CA-PINN uses

1. logarithmic densities, $u_n = \log n$ and $u_I = \log I$, so that both densities stay positive,
2. trial solutions with $d(x) = x(1-x)$ that satisfy the Dirichlet conditions exactly,
3. a soft zero-flux condition for the ions at both contacts,
4. a global ion-conservation constraint, $\int_0^1 I\,dx = P_0/N_{ref}$.

See Section 2 of the paper for the full formulation.

## Requirements

The results in the paper were computed on Google Colab (NVIDIA T4 GPU) with TensorFlow 2.20.0.

```bash
pip install -r requirements.txt
```

Training one CA-PINN takes about 25 minutes on a T4 GPU. Re-plotting from the saved results also works on a CPU.

## Repository structure

```
.
├── CA_PINN_Solver.ipynb            # CA-PINN solver
├── FD_Solver.ipynb                 # finite-difference reference solver
├── validation_utils.py             # relative L2 errors and PINN-vs-FD plots
│
├── 01_FD_Grid_Independence.ipynb   # FD grid refinement (250, 500, 1000 intervals)
├── 02_Architecture_Study.ipynb     # Tables 3-4: network depth and width
├── 03_Activation_Study.ipynb       # Table 5: activation functions
├── 04_Adam_Epoch_Study.ipynb       # Table 6: number of Adam epochs
├── 05_Validation.ipynb             # Table 7, Figs. 3-4: validation against FD
├── 06a_Ablation_Direct.ipynb       # Table 8: Direct PINN
├── 06b_Ablation_Direct_Cons.ipynb  # Table 8: Direct + Cons.
├── 06c_Ablation_Log_Density.ipynb  # Table 8: Log-Density PINN
├── 06d_Ablation_CA_PINN.ipynb      # Table 8: CA-PINN
├── 06e_Ablation_Plot.ipynb         # Fig. 5: ablation profiles
├── 07_P0_Sweep.ipynb               # Table 9, Fig. 6: mobile-ion density sweep
├── 08_Vi_Sweep.ipynb               # Table 10, Fig. 7: pre-bias voltage sweep
├── 09_Band_Diagrams.ipynb          # Fig. 8: energy-band diagrams
│
├── results/                        # saved results for all tables and figures
├── CITATION.cff
├── requirements.txt
└── LICENSE
```

`CA_PINN_Solver.ipynb` and `FD_Solver.ipynb` contain the two solvers. The numbered notebooks set the case parameters ($P_0$, $V_i$, seed, network, density representation, conservation on or off), run the solvers with `%run -i` and compare the results using `validation_utils.py`, so the solver notebooks do not need to be opened directly. If a notebook does not set $V_i$, both solvers use $V_i = 0.001$ V, which is the validation case in the paper.

Each CA-PINN run saves its solution profiles, the loss history (`loss_history.csv`) and the optimizer statistics (`training_metadata.csv`: Adam and L-BFGS run times, number of L-BFGS iterations and objective evaluations, convergence flag). The last file is used by `04_Adam_Epoch_Study` for Table 6.

## Running the notebooks

Locally:

```bash
git clone https://github.com/ammarsani/CA-PINN-SteadyState-DDP.git
cd CA-PINN-SteadyState-DDP
pip install -r requirements.txt
jupyter lab
```

On Google Colab (GPU runtime):

```python
!git clone https://github.com/ammarsani/CA-PINN-SteadyState-DDP.git
%cd CA-PINN-SteadyState-DDP
```

The notebooks use paths relative to the repository folder, so run them from there.

### Re-plotting and retraining

Each numbered notebook has switches at the top. Notebooks 02-05, 07 and 08 are set to re-plot: they rebuild the tables and figures from the saved results in a few minutes without training. To retrain, change the switches as follows.

| Notebooks | Switches for retraining |
|---|---|
| 02, 03, 04 | `RUN_FD_REFERENCE = True`, `RUN_PINN_CASES = True`, `OVERWRITE_EXISTING = True` |
| 05, 07, 08 | `RUN_PREBIAS = True`, `RUN_FD = True`, `RUN_COMPARE = True` |

The ablation notebooks 06a-06d are set to retrain, because their summary tables record the training time. Their results are already saved, so they only need to be run for retraining. Notebook 01 has no switches and reruns the FD solver (a few minutes); 06e and 09 only plot.

Approximate training times on a T4 GPU:

| Notebook | Runs | Time |
|---|---|---|
| 02_Architecture_Study | 11 architectures x 3 seeds | about 15 h |
| 03_Activation_Study | 5 activations x 3 seeds | about 5.5 h |
| 04_Adam_Epoch_Study | 3 settings x 3 seeds | about 4.5 h |
| 05_Validation | 3 seeds | about 1.3 h |
| 06a-06d (each) | 3 seeds | 1.3-1.6 h |
| 07_P0_Sweep | 5 values of $P_0$ | about 2.5 h |
| 08_Vi_Sweep | 5 values of $V_i$ | about 40 min each |

## Results in the paper

| Result | Notebook | Output in `results/` |
|---|---|---|
| FD grid refinement (Sec. 2.3) | 01_FD_Grid_Independence | `fd_grid_independence/grid_independence_summary.csv` |
| Tables 3-4 | 02_Architecture_Study | `architecture/architecture_mean_std.csv` |
| Table 5 | 03_Activation_Study | `activation/activation_mean_std.csv` |
| Table 6 | 04_Adam_Epoch_Study | `adam_iterations/adam_iteration_publication_table.csv` |
| Table 7, Fig. 3 | 05_Validation | `validation/validation_publication_table.csv`, `validation/validation_with_spatial_errors.pdf` |
| Fig. 4 | 05_Validation (seed 2024) | `validation/P0_1e+21_phi0.10_tanh_4x40_adam2000_lbfgs10000_seed2024/loss_history.pdf` |
| Table 8 | 06a-06d | `ablation_*/ablation_*_mean_std_summary.csv` |
| Fig. 5 | 06e_Ablation_Plot | `ablation_comparison/ablation_solution_profiles_3x4.pdf` |
| Table 9, Fig. 6 | 07_P0_Sweep | `P0_sweep/P0_sweep_summary.csv`, `P0_sweep/effect_of_P0_profiles_overlay.pdf` |
| Table 10, Fig. 7 | 08_Vi_Sweep | `Vi_sweep/Vi_sweep_summary.csv`, `Vi_sweep/effect_of_prebias_voltage_profiles_overlay.pdf` |
| Fig. 8 | 09_Band_Diagrams | `Vi_sweep/band_diagrams/energy_m0p10V.pdf`, `energy_p0p00V.pdf`, `energy_p0p25V.pdf` |

Each run folder in `results/` contains the PINN and FD profiles (`pinn_psi.csv`, `pinn_n.csv`, `pinn_I.csv`, `fd_*.csv`), the error metrics (`error_metrics.csv`) and the loss history (`loss_history.csv`).

## Notes

- Tables 3-5 were computed with 5000 Adam epochs. Table 6 compares 2000, 5000 and 10000 Adam epochs, and Tables 7-10 use the selected value of 2000. All runs use 10000 L-BFGS iterations after Adam.
- The ablation runs (Table 8) were trained independently of the validation runs (Table 7). Training on a GPU is not exactly reproducible even with fixed seeds, so repeated runs give slightly different errors; in all repeated CA-PINN runs the relative L2 errors stayed below 1%.
- The training-history plots record the L-BFGS stage once per objective-function evaluation (about 28,800 evaluations for 10000 iterations).
- The FD solver uses an under-relaxation factor of 0.01 and at most 5000 Gummel iterations. For $P_0 \ge 10^{24}$ m⁻³ it uses 0.001 and 70000 iterations (Section 3.4).
- Fig. 8 uses $\chi = 4.0$ eV and $E_g = 1.58$ eV. The notebook plots $E_c$, $E_{Fn}$ and $E_v$; the panel labels and the schematic ion regions and arrows in the paper figure were added afterwards.
- `results/` contains only the cases reported in the paper. Network weights (`.h5`) and PNG copies of the figures are not included.

## Default settings

| Item | Value |
|---|---|
| Network | 4 hidden layers x 40 neurons, tanh, linear output layer |
| Optimizer | Adam (2000 epochs, learning rate $10^{-3}$ halved every 1000 steps), then L-BFGS (10000 iterations) |
| Collocation points | 99 interior points (tanh-stretched mesh) and 2 boundary points |
| Loss weights | $W_\psi = W_n = W_I = W_{FI} = 1$, $W_{cons} = 100$ |
| Seeds | 42, 1234, 2024 |
| Device | $b = 500$ nm, $N_c = 2.1\times10^{24}$ m⁻³, $\mu_n = 10^{-4}$ m² V⁻¹ s⁻¹, $P_0 = 10^{21}$ m⁻³, $\varepsilon_r = 50$, $T = 300$ K, $\phi_a = \phi_c = 0.1$ eV, $V_{bi} = 0$, $V_i = 0.001$ V |

## Citation

Please cite the paper above if you use this code (see also `CITATION.cff`). The full reference will be added after publication.

## License

MIT License, see `LICENSE`.
