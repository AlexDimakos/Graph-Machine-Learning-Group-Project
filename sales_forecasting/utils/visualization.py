import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Fira Sans'

def plot_t_spectral_energy(lam, X_hat, t):
    xhat_t = X_hat[t]                                  
    plt.figure()
    plt.grid()
    plt.stem(lam, np.abs(xhat_t), basefmt=" ")
    plt.xlabel("Graph frequency (eigenvalue λ_k)")
    plt.ylabel(r"Amplitude $|\hat{x}_k|$")
    plt.title(f"Graph Spectrum of Sales (t={t})")
    plt.tight_layout(); plt.show()


def plot_avg_spectral_energy(lam, X_hat):
    avg_amp = np.mean(np.abs(X_hat), axis=0)              # mean over time
    plt.figure(figsize=(6,4))
    plt.stem(lam, avg_amp, basefmt=" ")
    plt.xlabel("Graph frequency (eigenvalue λ_k)")
    plt.ylabel(r"Avg amplitude $\mathbb{E}[|\hat{x}_k|]$")
    plt.title("Average Graph Spectrum of Sales (training window)")
    plt.tight_layout(); plt.show()
    

def plot_heatmap_spectral_energy(lam, X_hat, precision=2):
    # sort by eigenvalue
    idx = np.argsort(lam)
    lam_sorted = lam[idx]
    X_sorted = np.abs(X_hat[:, idx]).T  # (N, T)

    # round eigenvalues to group similar ones
    lam_rounded = np.round(lam_sorted, decimals=precision)

    # find unique groups
    unique_vals = np.unique(lam_rounded)

    # average energy per group
    grouped_data = []
    for val in unique_vals:
        mask = lam_rounded == val
        grouped_data.append(X_sorted[mask].mean(axis=0))  # average across rows with same eigenvalue

    grouped_data = np.array(grouped_data)  # (num_groups, T)

    # plotting
    plt.figure(plt.figure(figsize=(10,4)))
    plt.imshow(grouped_data, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='x̂ₖ(t)')
    plt.yticks(np.arange(len(unique_vals)), [f'{v:.{precision}f}' for v in unique_vals])
    plt.xlabel('Time step t')
    plt.ylabel('Graph frequency')
    plt.title('Graph Spectrogram')
    plt.tight_layout()
    plt.show()


