"""
Quantum well (1D) — Finite difference vs Numerov
Produces 6 plots:
  - Top row: 3 wavefunctions from the finite-difference ("well") solver
             (first two even states and the first odd state when available)
  - Bottom row: 3 wavefunctions from the Numerov solver (first 3 states)

All code in English and documented.
Author: (your name)
"""

import numpy as np
import matplotlib.pyplot as plt


# --------------------------
# Physical constants & grid
# --------------------------
V0 = 244.0                 # eV (depth)
a = 1e-10                  # m (half width reference in code)
c = 3e8                    # m/s
me = 0.511e6 / c**2        # (kept as in the original code's units)
hbar = 6.582e-16           # eV*s (reduced Planck's constant)

k = (2 * me * a**2 * V0) / hbar**2

# Grid parameters (kept as in the original)
umin, umax = -2 * a, 2 * a
npts = 1000                # base grid size used in original
nreps = 200
u = np.linspace(umin, umax, npts) / a    # dimensionless coordinate (used for FD)
du = u[1] - u[0]

alpha_init = np.pi**2 / k
dalpha = alpha_init / 10.0


# --------------------------
# Potential-like coefficient
# --------------------------
def C(u_val, alpha):
    """
    Returns the coefficient C(u, alpha) used in the differential equation.
    Matches the conditional in the original script.
    """
    return -k * alpha if (-0.5 < u_val < 0.5) else k * (1.0 - alpha)


# --------------------------
# Finite-difference utilities
# --------------------------
def fd_end_values(alpha):
    """
    Integrate from center (half-grid) using simple Euler-like step as in original
    and return the *last* values for even and odd parity solutions.
    This function corresponds to the 'PSI' behavior in the original code.
    """
    half = npts // 2
    psi_even = np.zeros(half)
    psi_odd = np.zeros(half)
    dpsi_even = np.zeros(half)
    dpsi_odd = np.zeros(half)

    # boundary conditions (center -> outward)
    psi_even[0], dpsi_even[0] = 1.0, 0.0
    psi_odd[0], dpsi_odd[0] = 0.0, 1.0

    for i in range(1, half):
        psi_even[i] = psi_even[i - 1] + dpsi_even[i - 1] * du
        dpsi_even[i] = (
            dpsi_even[i - 1]
            + C(u[i + half], alpha) * psi_even[i] * du
        )

        psi_odd[i] = psi_odd[i - 1] + dpsi_odd[i - 1] * du
        dpsi_odd[i] = (
            dpsi_odd[i - 1]
            + C(u[i + half], alpha) * psi_odd[i] * du
        )

    return psi_even[-1], psi_odd[-1]


def bisect_fd(alpha0, nmax, parity):
    """
    Bisection search for FD roots.
    parity: 0 -> even, 1 -> odd
    Returns up to nmax alpha roots (as numpy array).
    """
    sols = []
    a1 = alpha0
    a2 = alpha0 + dalpha

    while len(sols) < nmax:
        # move interval until sign change or alpha cap
        while (
            fd_end_values(a1)[parity] * fd_end_values(a2)[parity] > 0
            and a2 < 1.0
        ):
            a1 = a2
            a2 += dalpha

        # if we've gone beyond reasonable alpha, break
        if a2 >= 1.0 and fd_end_values(a1)[parity] * fd_end_values(a2)[parity] > 0:
            break

        # bisection refine
        for _ in range(nreps):
            amid = 0.5 * (a1 + a2)
            if fd_end_values(a1)[parity] * fd_end_values(amid)[parity] < 0:
                a2 = amid
            else:
                a1 = amid

        root = 0.5 * (a1 + a2)
        if root > 1.0:
            break

        sols.append(root)
        # advance search window
        a1 = a2
        a2 = a1 + dalpha

    return np.array(sols)


def wavefunction_fd(alpha, parity="even"):
    """
    Build the full (symmetric) FD wavefunction for plotting.
    Returns a vector of length npts (same x sampling as `u`).
    """
    half = npts // 2
    psi = np.zeros(half)
    dpsi = np.zeros(half)

    if parity == "even":
        psi[0], dpsi[0] = 1.0, 0.0
    else:
        psi[0], dpsi[0] = 0.0, 1.0

    for i in range(1, half):
        psi[i] = psi[i - 1] + dpsi[i - 1] * du
        dpsi[i] = dpsi[i - 1] + C(u[i + half], alpha) * psi[i] * du

    if parity == "even":
        full = np.concatenate((psi[::-1], psi))
    else:
        full = np.concatenate((-psi[::-1], psi))

    # normalize
    norm = np.linalg.norm(full)
    if norm == 0:
        return full
    return full / norm


# --------------------------
# Numerov utilities
# --------------------------
def numerov_last(alpha):
    """
    Run the Numerov-type integration as in the original code and return psi[-1].
    Used for root searching.
    """
    psi = np.zeros(npts)
    phi = np.zeros(npts)

    # initial conditions (kept from original)
    psi[0], psi[1] = 1e-2, 1e-1
    phi[0] = psi[0] * (1 - du**2 * C(u[0], alpha) / 12.0)
    phi[1] = psi[1] * (1 - du**2 * C(u[1], alpha) / 12.0)

    for i in range(1, npts - 1):
        phi[i + 1] = 2.0 * phi[i] - phi[i - 1] + du**2 * C(u[i], alpha) * psi[i]
        psi[i + 1] = phi[i + 1] / (1.0 - du**2 * C(u[i + 1], alpha) / 12.0)

    return psi[-1]


def bisect_numerov(alpha0, nmax):
    """
    Bisection search using numerov_last() as the sign function.
    Returns up to nmax alpha roots.
    """
    sols = []
    a1 = alpha0
    a2 = alpha0 + dalpha

    while len(sols) < nmax:
        while numerov_last(a1) * numerov_last(a2) > 0 and a2 < 1.0:
            a1 = a2
            a2 += dalpha

        if a2 >= 1.0 and numerov_last(a1) * numerov_last(a2) > 0:
            break

        for _ in range(nreps):
            amid = 0.5 * (a1 + a2)
            if numerov_last(a1) * numerov_last(amid) < 0:
                a2 = amid
            else:
                a1 = amid

        root = 0.5 * (a1 + a2)
        if root > 1.0:
            break

        sols.append(root)
        a1 = a2
        a2 = a1 + dalpha

    return np.array(sols)


def wavefunction_numerov(alpha):
    """
    Build the Numerov wavefunction for plotting.
    Returns a full, symmetric wavefunction of length 2*npts (as in original script).
    """
    psi = np.zeros(npts)
    phi = np.zeros(npts)

    psi[0], psi[1] = 1e-2, 1e-1
    phi[0] = psi[0] * (1 - du**2 * C(u[0], alpha) / 12.0)
    phi[1] = psi[1] * (1 - du**2 * C(u[1], alpha) / 12.0)

    for i in range(1, npts - 1):
        phi[i + 1] = 2.0 * phi[i] - phi[i - 1] + du**2 * C(u[i], alpha) * psi[i]
        psi[i + 1] = phi[i + 1] / (1.0 - du**2 * C(u[i + 1], alpha) / 12.0)

    full = np.concatenate((psi[::-1], psi))
    norm = np.linalg.norm(full)
    return full / norm if norm != 0 else full


# --------------------------
# Main: compute roots and plot
# --------------------------
def main():
    nmax = 4  # ask for up to 4 roots, like original

    # theoretical alphas (kept from original)
    alpha_theory = [0.09797, 0.3825, 0.8075]
    E_theory = [alpha * V0 for alpha in alpha_theory]

    # find finite-difference roots: even and odd
    alpha_even = bisect_fd(0.0, nmax, parity=0)
    alpha_odd = bisect_fd(0.0, nmax, parity=1)

    # Construct the three 'well' alphas like in the original:
    # [first even, second even, first odd] if available
    alphas_well = []
    alphas_well.append(float(alpha_even[0]) if alpha_even.size >= 1 else None)
    alphas_well.append(float(alpha_even[1]) if alpha_even.size >= 2 else None)
    alphas_well.append(float(alpha_odd[0]) if alpha_odd.size >= 1 else None)

    # Numerov roots (up to 3 we will plot)
    alpha_num = bisect_numerov(0.0, nmax)

    # Print energies (careful with availability)
    def to_energies(alpha_list):
        return [alpha * V0 for alpha in alpha_list if alpha is not None]

    print("=== Theoretical alphas and energies (kept for reference) ===")
    for a_th, E_th in zip(alpha_theory, E_theory):
        print(f" alpha={a_th:.6f}  E={E_th:.6f} eV")

    print("\n=== Finite-difference (well) alphas found ===")
    print(" even roots:", alpha_even)
    print(" odd roots :", alpha_odd)

    print("\nConstructed 'well' alphas (first-even, second-even, first-odd):")
    print(alphas_well)

    print("\n=== Numerov alphas found ===")
    print(alpha_num)

    # energies
    print("\nEnergies (FD well):", to_energies(alphas_well))
    print("Energies (Numerov):", [alpha * V0 for alpha in alpha_num])

    # ---------------------
    # Plot: 2 rows x 3 columns -> 6 plots total
    # ---------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)

    # Top row: finite-difference "well" wavefunctions
    for i in range(3):
        ax = axes[0, i]
        alpha = alphas_well[i] if i < len(alphas_well) else None
        if alpha is None:
            ax.text(0.5, 0.5, "no root\nfound", ha="center", va="center", fontsize=12)
            ax.axis("off")
            continue

        parity = "even" if i < 2 else "odd"
        psi_fd = wavefunction_fd(alpha, parity=parity)
        ax.plot(u, psi_fd, label=f"FD {parity.capitalize()} α={alpha:.6f}")
        ax.set_xlim(u[0], u[-1])
        ax.grid(True)
        ax.legend(fontsize=9)
        ax.set_xlabel("u / a")
        ax.set_ylabel("ψ (normalized)")
        ax.set_title(f"FD — {parity.capitalize()} state")

    # Bottom row: Numerov wavefunctions (first 3 numerov roots)
    u_num = np.linspace(umin, umax, 2 * npts) / a
    for i in range(3):
        ax = axes[1, i]
        if i >= alpha_num.size:
            ax.text(0.5, 0.5, "no root\nfound", ha="center", va="center", fontsize=12)
            ax.axis("off")
            continue

        alpha = float(alpha_num[i])
        psi_num = wavefunction_numerov(alpha)
        ax.plot(u_num, psi_num, label=f"Numerov α={alpha:.6f}", linestyle="-")
        ax.set_xlim(u_num[0], u_num[-1])
        ax.grid(True)
        ax.legend(fontsize=9)
        ax.set_xlabel("u / a")
        ax.set_ylabel("ψ (normalized)")
        ax.set_title(f"Numerov state #{i+1}")

    fig.suptitle("Finite Difference (top) vs Numerov (bottom) — wavefunctions", fontsize=14)
    plt.show()


if __name__ == "__main__":
    main()
