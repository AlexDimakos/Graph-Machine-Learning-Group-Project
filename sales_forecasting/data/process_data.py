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

    # Manually drop duplicate node
    production.drop(columns=["POP001L12P.1"], inplace=True)
    factory_issue.drop(columns=["POP001L12P.1"], inplace=True)
    delivery.drop(columns=["POP001L12P.1"], inplace=True)
    sales_order.drop(columns=["POP001L12P.1"], inplace=True)

    ## Filter nodes with very limited information
    products = [col for col in production.columns if col != "Date"]

    # Compute filtering stats per node (product) using pandas so we keep names
    cv_vals = {}
    zero_ratios = {}

    for prod in products:
        # Concatenate the four signals for this product over time so we can
        # compute mean/std/zero-ratio across all values for the product
        concat_series = pd.concat(
            [production[prod], factory_issue[prod], delivery[prod], sales_order[prod]]
        )
        mean = concat_series.mean()
        std = concat_series.std()
        cv = std / (abs(mean) + 1e-8)
        zr = (concat_series == 0).mean()

        cv_vals[prod] = cv
        zero_ratios[prod] = zr

    cv_thr = 0.01  # drop nodes with <1% relative variation
    zero_thr = 0.6  # drop nodes that are >=60% zeros

    keep_products = [
        prod
        for prod in products
        if (cv_vals[prod] > cv_thr and zero_ratios[prod] < zero_thr)
    ]

    removed = len(products) - len(keep_products)
    print(f"Removed {removed} of {len(products)} nodes ({removed / len(products):.1%})")

    # Filter the original DataFrames by column names (this keeps product names)
    X_prod = production[keep_products]
    X_issue = factory_issue[keep_products]
    X_delivery = delivery[keep_products]
    X_sales = sales_order[keep_products]

    # Convert to numpy arrays and stack features as before
    X_prod_np = X_prod.to_numpy()
    X_issue_np = X_issue.to_numpy()
    X_delivery_np = X_delivery.to_numpy()
    X_sales_np = X_sales.to_numpy()

    # shape: [T, N, F] = [time_steps, num_nodes, num_features]
    X = np.stack([X_prod_np, X_issue_np, X_delivery_np, X_sales_np], axis=-1)

    X_sales_np = X[:, :, 3]

    # Matching the features at a timestep with their label (the next timestep)
    y = X_sales_np[1:]
    X = X[:-1]

    np.save(PROCESSED_DATA_DIR / "X", X)
    np.save(PROCESSED_DATA_DIR / "y", y)

    ## Transform the edges, filtering out the ones incident to nodes that were removed
    edge_list = [edges_plant, edges_group, edges_subgroup, edges_storage]
    edge_names = ["edges_plant", "edges_group", "edges_subgroup", "edges_storage"]
    for edges, name in zip(edge_list, edge_names):
        edges["node1"] = edges["node1"].astype(str)
        edges["node2"] = edges["node2"].astype(str)

        # Keep only edges where both endpoints are in the filtered product list
        # (we kept original product names in `keep_products`)
        edges = edges[
            (edges["node1"].isin(keep_products)) & (edges["node2"].isin(keep_products))
        ]

        # Now map product names to new contiguous ids based on filtered order
        product_to_id_filtered = {prod: i for i, prod in enumerate(keep_products)}
        edges.loc[:, "node1"] = edges.loc[:, "node1"].map(product_to_id_filtered)
        edges.loc[:, "node2"] = edges.loc[:, "node2"].map(product_to_id_filtered)

        # ! There are pairs of the same nodes being connected by multiple edges of the same type (e.g. multiple plants)
        # Group by node pairs and sum the quantities
        edges = edges.copy()
        edges["pair"] = edges[["node1", "node2"]].apply(
            lambda x: tuple(sorted(x)), axis=1
        )
        edges = edges.groupby("pair").size().reset_index(name="multiplicity")
        edges[["node1", "node2"]] = pd.DataFrame(
            edges["pair"].tolist(), index=edges.index
        )
        edges.drop(columns="pair", inplace=True)
        # Just reorder to have the edge first and then the multiplicity
        edges = edges[["node1", "node2", "multiplicity"]]

        unique_edges_count = edges.shape[0]
        print(f"Number of unique edges (node1, node2): {unique_edges_count}")

        edge_index = edges[["node1", "node2"]].to_numpy().T.astype(np.int64)
        np.save(PROCESSED_DATA_DIR / f"{name}", edge_index)

        edge_index_weighted = edges.to_numpy().T.astype(np.int64)
        np.save(PROCESSED_DATA_DIR / f"{name}_weighted", edge_index_weighted)


if __name__ == "__main__":
    main()
