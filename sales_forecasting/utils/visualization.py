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
    

def plot_heatmap_spectral_energy(lam, X_hat):
    A = np.abs(X_hat)                   # (T, N)
    idx = np.argsort(lam)               # sort by eigenvalue
    lam_sorted = lam[idx]
    A_sorted = A[:, idx].T              # (N, T)  

    plt.figure(figsize=(10,6))
    plt.imshow(A_sorted, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='|x̂ₖ(t)|')

    # one row per eigenmode; label rows by λ_k
    N = len(lam_sorted)
    plt.yticks(np.arange(N), [f'{l:.2f}' for l in lam_sorted])
    plt.xlabel('Time step t')
    plt.ylabel('Graph frequency λₖ')
    plt.title('Graph Spectrogram')
    plt.tight_layout()
    plt.show()

