from pathlib import Path

import numpy as np
import pandas as pd

from sales_forecasting.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


def transform_temporal(df, mapping):
    df_no_date = df.drop(columns=["Date"])
    return df_no_date.rename(columns=mapping)


def main():
    ## Load data
    edges_plant = pd.read_csv(RAW_DATA_DIR / Path("Edges/Edges (Plant).csv"))
    edges_group = pd.read_csv(RAW_DATA_DIR / Path("Edges/Edges (Product Group).csv"))
    edges_subgroup = pd.read_csv(
        RAW_DATA_DIR / Path("Edges/Edges (Product Sub-Group).csv")
    )
    edges_storage = pd.read_csv(
        RAW_DATA_DIR / Path("Edges/Edges (Storage Location).csv")
    )

    signals_path = RAW_DATA_DIR / Path("Temporal Data/Unit/")
    production = pd.read_csv(signals_path / Path("Production .csv"))
    factory_issue = pd.read_csv(signals_path / Path("Factory Issue.csv"))
    delivery = pd.read_csv(signals_path / Path("Delivery To distributor.csv"))
    sales_order = pd.read_csv(signals_path / Path("Sales Order.csv"))

    production.drop(columns=["POP001L12P.1"], inplace=True)
    delivery.drop(columns=["POP001L12P.1"], inplace=True)
    sales_order.drop(columns=["POP001L12P.1"], inplace=True)
    factory_issue.drop(columns=["POP001L12P.1"], inplace=True)

    ## Transform the dataframes
    products = [col for col in production.columns if col != "Date"]
    product_to_id = {prod: i for i, prod in enumerate(products)}

    X_prod = transform_temporal(production, product_to_id)
    X_issue = transform_temporal(factory_issue, product_to_id)
    X_delivery = transform_temporal(delivery, product_to_id)
    X_sales = transform_temporal(sales_order, product_to_id)

    X_prod_np = X_prod.to_numpy()
    X_issue_np = X_issue.to_numpy()
    X_delivery_np = X_delivery.to_numpy()
    X_sales_np = X_sales.to_numpy()

    # shape: [T, N, F] = [time_steps, num_nodes, num_features]
    X = np.stack([X_prod_np, X_issue_np, X_delivery_np, X_sales_np], axis=-1)

    # (a) Mean and std over time + features
    mean_per_node = X.mean(axis=(0, 2))
    std_per_node  = X.std(axis=(0, 2))

    # (b) Coefficient of variation (normalized variability)
    cv_per_node = np.divide(std_per_node, np.abs(mean_per_node) + 1e-8)

    # (c) Zero ratio
    zero_ratio = (X == 0).mean(axis=(0, 2))

    cv_thr = 0.01        # drop nodes with <1% relative variation
    zero_thr = 0.6      # drop nodes that are ≥95% zeros
    mask = (cv_per_node > cv_thr) & (zero_ratio < zero_thr)

    X_filtered = X[:, mask, :]
    
    print(f"Removed {(~mask).sum()} of {len(mask)} nodes "f"({(~mask).sum()/len(mask):.1%})")
    
    X = X_filtered

    X_sales_np = X[:, :, 3]

    # Matching the features at a timestep with their label (the next timestep)
    y = X_sales_np[1:]
    X = X[:-1]

    np.save(PROCESSED_DATA_DIR / "X", X)
    np.save(PROCESSED_DATA_DIR / "y", y)

    ## Transform the edges
    edge_list = [edges_plant, edges_group, edges_subgroup, edges_storage]
    edge_names = ["edges_plant", "edges_group", "edges_subgroup", "edges_storage"]
    for edges, name in zip(edge_list, edge_names):
        edges["node1"] = edges["node1"].astype(str)
        edges["node2"] = edges["node2"].astype(str)

        edges["node1"] = edges["node1"].map(product_to_id)
        edges["node2"] = edges["node2"].map(product_to_id)

        edge_index = edges[["node1", "node2"]].to_numpy().T.astype(np.int64)
        np.save(PROCESSED_DATA_DIR / f"{name}", edge_index)


if __name__ == "__main__":
    main()
