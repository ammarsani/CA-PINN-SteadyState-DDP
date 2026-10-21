import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def rel_l2(pinn, ref, eps=1e-300):
    pinn = np.asarray(pinn, dtype=float)
    ref  = np.asarray(ref, dtype=float)
    return np.linalg.norm(pinn - ref) / (np.linalg.norm(ref) + eps)

def load_pair(base_dir="outputs"):
    p_psi = pd.read_csv(os.path.join(base_dir, "pinn_psi.csv"))
    p_n   = pd.read_csv(os.path.join(base_dir, "pinn_n.csv"))
    p_I   = pd.read_csv(os.path.join(base_dir, "pinn_I.csv"))

    f_psi = pd.read_csv(os.path.join(base_dir, "fd_psi.csv"))
    f_n   = pd.read_csv(os.path.join(base_dir, "fd_n.csv"))
    f_I   = pd.read_csv(os.path.join(base_dir, "fd_I.csv"))

    return (p_psi, p_n, p_I), (f_psi, f_n, f_I)

def check_same_grid(p, f, name, tol_nm=1e-9):
    if len(p) != len(f):
        raise ValueError(f"{name}: PINN and FD lengths differ.")

    dx = np.max(np.abs(np.asarray(p["x_nm"]) - np.asarray(f["x_nm"])))
    if dx > tol_nm:
        raise ValueError(f"{name}: PINN and FD x-grids differ. Max |Δx| = {dx:.3e} nm.")

def compare_and_plot(base_dir="outputs", title_suffix=""):
    (p_psi, p_n, p_I), (f_psi, f_n, f_I) = load_pair(base_dir)

    check_same_grid(p_psi, f_psi, "psi")
    check_same_grid(p_n,   f_n,   "n")
    check_same_grid(p_I,   f_I,   "I")

    metrics = {
        "relL2_psi": rel_l2(p_psi["psi_V"], f_psi["psi_V"]),
        "relL2_n":   rel_l2(p_n["n_m3"],   f_n["n_m3"]),
        "relL2_I":   rel_l2(p_I["I_m3"],   f_I["I_m3"]),
        "I_mean_PINN": float(np.mean(p_I["I_m3"])),
        "I_mean_FD":   float(np.mean(f_I["I_m3"])),
    }

    metrics["I_mean_rel_error"] = abs(
        metrics["I_mean_PINN"] - metrics["I_mean_FD"]
    ) / (abs(metrics["I_mean_FD"]) + 1e-300)

    df = pd.DataFrame([metrics])
    df.to_csv(os.path.join(base_dir, "error_metrics.csv"), index=False)

    print(f"[PreBias validation] {title_suffix}".strip())
    print(f"relL2(psi) = {metrics['relL2_psi']:.3e}")
    print(f"relL2(n)   = {metrics['relL2_n']:.3e}")
    print(f"relL2(I)   = {metrics['relL2_I']:.3e}")
    print(f"Mean I PINN = {metrics['I_mean_PINN']:.3e} m^-3")
    print(f"Mean I FD   = {metrics['I_mean_FD']:.3e} m^-3")
    print(f"Mean I rel. error = {metrics['I_mean_rel_error']:.3e}")

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))

    ax[0].plot(f_psi["x_nm"], f_psi["psi_V"], label="FD", linewidth=2)
    ax[0].plot(p_psi["x_nm"], p_psi["psi_V"], "--", label="PINN", linewidth=2)
    ax[0].set_title("(a)")
    ax[0].set_ylabel(r"$\psi$ (V)")

    ax[1].semilogy(f_n["x_nm"], f_n["n_m3"], label="FD", linewidth=2)
    ax[1].semilogy(p_n["x_nm"], p_n["n_m3"], "--", label="PINN", linewidth=2)
    ax[1].set_title("(b)")
    ax[1].set_ylabel(r"$n$ (m$^{-3}$)")

    ax[2].semilogy(f_I["x_nm"], f_I["I_m3"], label="FD", linewidth=2)
    ax[2].semilogy(p_I["x_nm"], p_I["I_m3"], "--", label="PINN", linewidth=2)
    ax[2].set_title("(c)")
    ax[2].set_ylabel(r"$I$ (m$^{-3}$)")

    for a in ax:
        a.set_xlabel("Position (nm)")
        a.grid(True, which="both", alpha=0.25)
        a.legend(frameon=False)

    plt.tight_layout()

    png_path = os.path.join(base_dir, "pinn_vs_fd_profiles.png")
    pdf_path = os.path.join(base_dir, "pinn_vs_fd_profiles.pdf")

    plt.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.show()

    print("Saved:", png_path)
    print("Saved:", pdf_path)

    return metrics
    
    
def poisson_diagnostics(base_dir, lambda_sq, N_ref, Vt, title_suffix=""):
    psi_df = pd.read_csv(os.path.join(base_dir, "pinn_psi.csv"))
    n_df   = pd.read_csv(os.path.join(base_dir, "pinn_n.csv"))
    I_df   = pd.read_csv(os.path.join(base_dir, "pinn_I.csv"))

    # ---------------- Common grid ----------------
    x_nm = psi_df["x_nm"].values

    psi_V = psi_df["psi_V"].values
    n_m3  = np.interp(x_nm, n_df["x_nm"].values, n_df["n_m3"].values)
    I_m3  = np.interp(x_nm, I_df["x_nm"].values, I_df["I_m3"].values)

    x_nd = x_nm / np.max(x_nm)

    psi = psi_V / Vt
    n   = n_m3 / N_ref
    I   = I_m3 / N_ref

    # --------------------------------------------------
    # Second derivative on nonuniform dimensionless grid
    # --------------------------------------------------
    psi_xx = np.zeros_like(psi)
    
    dxm = x_nd[1:-1] - x_nd[:-2]
    dxp = x_nd[2:]   - x_nd[1:-1]
    
    psi_xx[1:-1] = 2.0 * (
        dxm * psi[2:] - (dxm + dxp) * psi[1:-1] + dxp * psi[:-2]
    ) / (dxm * dxp * (dxm + dxp))
    
    # Copy nearest interior values to boundaries
    psi_xx[0]  = psi_xx[1]
    psi_xx[-1] = psi_xx[-2]

    curvature = lambda_sq * psi_xx
    charge = I - n
    Rpsi = curvature + charge

    diag = pd.DataFrame({
        "x_nm": x_nm,
        "x_nd": x_nd,
        "lambda2_psi_xx": curvature,
        "I_minus_n": charge,
        "R_psi": Rpsi,
        "abs_R_psi": np.abs(Rpsi)
    })

    diag_path = os.path.join(base_dir, "poisson_diagnostics.csv")
    diag.to_csv(diag_path, index=False)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    ax.plot(x_nm, Rpsi, label=r"$R_\psi$", linewidth=2)
    ax.plot(x_nm, curvature, "--", label=r"$\lambda^2\psi_{xx}$", linewidth=2)
    ax.plot(x_nm, charge, ":", label=r"$I-n$", linewidth=2)

    ax.set_xlabel("Position (nm)")
    ax.set_ylabel("Dimensionless value")
    ax.set_title(title_suffix)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    plt.tight_layout()

    png_path = os.path.join(base_dir, "poisson_diagnostics.png")
    pdf_path = os.path.join(base_dir, "poisson_diagnostics.pdf")

    plt.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.show()

    summary = {
        "max_abs_Rpsi": float(np.max(np.abs(Rpsi))),
        "mean_abs_Rpsi": float(np.mean(np.abs(Rpsi))),
        "max_abs_lambda2_psi_xx": float(np.max(np.abs(curvature))),
        "mean_abs_lambda2_psi_xx": float(np.mean(np.abs(curvature))),
        "max_abs_I_minus_n": float(np.max(np.abs(charge))),
        "mean_abs_I_minus_n": float(np.mean(np.abs(charge))),
    }

    summary_path = os.path.join(base_dir, "poisson_diagnostics_summary.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print(f"[Poisson diagnostics] {title_suffix}".strip())
    print(f"max |Rpsi|              = {summary['max_abs_Rpsi']:.3e}")
    print(f"mean |Rpsi|             = {summary['mean_abs_Rpsi']:.3e}")
    print(f"max |lambda^2 psi_xx|   = {summary['max_abs_lambda2_psi_xx']:.3e}")
    print(f"mean |lambda^2 psi_xx|  = {summary['mean_abs_lambda2_psi_xx']:.3e}")
    print(f"max |I-n|               = {summary['max_abs_I_minus_n']:.3e}")
    print(f"mean |I-n|              = {summary['mean_abs_I_minus_n']:.3e}")
    print("Saved:", diag_path)
    print("Saved:", png_path)
    print("Saved:", pdf_path)
    print("Saved:", summary_path)

    return diag, summary
    
    
def poisson_region_diagnostics(base_dir, title_suffix=""):
    diag_path = os.path.join(base_dir, "poisson_diagnostics.csv")
    diag = pd.read_csv(diag_path)

    x = diag["x_nd"].values
    abs_R = diag["abs_R_psi"].values

    regions = {
        "left_contact_0_0.05": x <= 0.05,
        "bulk_0.05_0.95": (x > 0.05) & (x < 0.95),
        "right_contact_0.95_1": x >= 0.95,
    }

    rows = []
    for name, mask in regions.items():
        rows.append({
            "case": title_suffix,
            "region": name,
            "num_points": int(np.sum(mask)),
            "mean_abs_Rpsi": float(np.mean(abs_R[mask])),
            "max_abs_Rpsi": float(np.max(abs_R[mask])),
        })

    df = pd.DataFrame(rows)

    save_path = os.path.join(base_dir, "poisson_region_diagnostics.csv")
    df.to_csv(save_path, index=False)

    print(f"[Poisson region diagnostics] {title_suffix}")
    print(df)
    print("Saved:", save_path)

    return df