import numpy as np
from sklearn.preprocessing import StandardScaler
from sales_forecasting.models.train import scale_per_product
from scipy.sparse import coo_matrix
from sklearn.model_selection import TimeSeriesSplit
import pygsp as pg
from sales_forecasting import config
from sales_forecasting.utils.visualization import plot_t_spectral_energy, plot_avg_spectral_energy, plot_heatmap_spectral_energy


def load_training_data():
    """Load processed data and return training split."""
    X = np.load(config.PROCESSED_DATA_DIR / "X.npy")
    y = np.load(config.PROCESSED_DATA_DIR / "y.npy")
    
    # print("X shape is: " + str(X.shape))

    # tscv = TimeSeriesSplit(n_splits=config.TrainingConfig.n_splits)
    # train_idx, _ = next(tscv.split(X))
    # X_train = X[train_idx]
    # print("X train shape is: " + str(X_train.shape))
    # y_train = y[train_idx]
    # return X_train, y_train
    #X_scaled, y_scaled, _, _, _, _ = scale_per_product(X, y)
    
    T, N, F = X.shape

    X_scaled = np.zeros_like(X)
    y_scaled = np.zeros_like(y)

    scaler_X_per_product = []
    scaler_y_per_product = []

    for node_idx in range(N):
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        # Extract product time series
        X_node = X[:, node_idx, :]          
        y_node = y[:, node_idx].reshape(-1, 1)

        # Fit and transform
        X_scaled[:, node_idx, :] = scaler_X.fit_transform(X_node)
        y_scaled[:, node_idx] = scaler_y.fit_transform(y_node).ravel()

        scaler_X_per_product.append(scaler_X)
        scaler_y_per_product.append(scaler_y)

    
    
    return X_scaled, y_scaled

def build_graph(edge_type=None, num_active_nodes=None):
    """Load the edge index, drop edges to inactive nodes, and return a PyGSP Graph."""
    edge_type = edge_type or config.EDGE_TYPE
    edge_path = config.PROCESSED_DATA_DIR / f"edges_{edge_type}.npy"
    edge_index = np.load(edge_path)

    N = int(edge_index.max()) + 1

    # Filter edges that reference unused nodes 
    if num_active_nodes is not None and num_active_nodes < N:
        mask = (edge_index[0] < num_active_nodes) & (edge_index[1] < num_active_nodes)
        removed = np.sum(~mask)
        edge_index = edge_index[:, mask]
        print(f"Removed {removed} edges that referenced inactive nodes.")
        N = num_active_nodes

    # Build sparse adjacency matrix
    row, col = edge_index
    W = coo_matrix((np.ones(row.size), (row, col)), shape=(N, N)).tocsr()
    W = ((W + W.T) > 0).astype(float).tocsr()  # symmetrize
    W.setdiag(0)
    W.eliminate_zeros()

    # Build PyGSP graph
    G = pg.graphs.Graph(W)
    G.compute_laplacian(lap_type="normalized")
    G.compute_fourier_basis()
    return G



def graph_fourier_transform(G, X_train, feature_index=0):
    """Compute the graph Fourier transform (GFT) for a selected feature over time."""
    X_tn = X_train[:, :, feature_index].astype(float)  # shape [T, N]
    #print("This is the shape: " + str(X_tn.shape))
    X_hat = G.gft(X_tn.T).T  # [T, N] spectral coefficients per timestep
    return X_hat

def run_spectral_analysis(feature_index=3, t=0, X_train = None, y_train = None):
    """Full pipeline for spectral analysis."""
    #print("Running spectral analysis...")
    # use provided data if given; otherwise load training split
    if X_train is None or y_train is None:
        X_train, y_train = load_training_data()
    
        
    num_active_nodes = X_train.shape[1]
    G = build_graph(edge_type = None, num_active_nodes=num_active_nodes)

    X_hat = graph_fourier_transform(G, X_train, feature_index=feature_index)

    #print("Graph Laplacian eigenvalues (λ):", G.e)
    #print("GFT coefficients shape:", X_hat.shape)

    plot_avg_spectral_energy(G.e, X_hat)
    plot_t_spectral_energy(G.e, X_hat, t)
    plot_heatmap_spectral_energy(G.e, X_hat)
    #print("Spectral analysis complete.")

    return G, X_hat


if __name__ == "__main__":
    run_spectral_analysis()