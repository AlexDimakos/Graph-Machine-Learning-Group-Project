import sys
from pathlib import Path

import numpy as np
import pandas as pd

# add parent directory of src to Python path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import *


def transform_temporal(df, mapping):
    df_no_date = df.drop(columns=["Date"])
    return df_no_date.rename(columns=mapping)


def main():
    ## Load data
    nodes = pd.read_csv(RAW_DATA_DIR / Path("Nodes/Nodes.csv"))
    edges = pd.read_csv(RAW_DATA_DIR / Path("Edges/Edges (Product Sub-Group).csv"))

    signals_path = RAW_DATA_DIR / Path("Temporal Data/Unit/")
    production = pd.read_csv(signals_path / Path("Production .csv"))
    factory_issue = pd.read_csv(signals_path / Path("factory issue.csv"))
    delivery = pd.read_csv(signals_path / Path("Delivery To distributor.csv"))
    sales_order = pd.read_csv(signals_path / Path("Sales order.csv"))

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

    # Matching the features at a timestep with their label (the next timestep)
    y = X_sales_np[1:]
    X = X[:-1]

    np.save(PROCESSED_DATA_DIR / "X", X)
    np.save(PROCESSED_DATA_DIR / "y", y)


if __name__ == "__main__":
    main()
