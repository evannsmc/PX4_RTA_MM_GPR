import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def trim_trailing_all_nan(X: np.ndarray) -> np.ndarray:
    if X.size == 0:
        return X
    keep = ~np.isnan(X).all(axis=1)    # True for rows that are not all-NaN
    if not keep.any():
        return X[:0]                   # all rows were NaN
    last_valid = np.max(np.where(keep))
    return X[:last_valid+1]

def get_flat_output_and_desired(df):
    # Extract actual and reference values as numpy arrays
    angle_col = 'psi' if 'psi' in df.columns else 'yaw'    # Pick either 'psi' or 'yaw'
    angle_ref_col = 'psi_ref' if 'psi_ref' in df.columns else 'yaw_ref'

    actual_values = df[['x', 'y', 'z', angle_col]].to_numpy()
    reference_values = df[['x_ref', 'y_ref', 'z_ref', angle_ref_col]].to_numpy()

    actual_values_clean = trim_trailing_all_nan(actual_values)
    reference_values_clean = trim_trailing_all_nan(reference_values)

    n = min(len(actual_values_clean), len(reference_values_clean))
    actual_values_clean = actual_values_clean[:n]
    reference_values_clean = reference_values_clean[:n]

    # Flip z and z_ref (3rd column, index 2)
    actual_values_clean[:, 2] *= -1
    reference_values_clean[:, 2] *= -1

    return actual_values_clean, reference_values_clean

def calculate_overall_rmse(df):
    """
    Calculate the overall RMSE across x, y, z, and yaw compared to their reference values.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing the actual values and reference values.
    
    Returns:
    float: The overall RMSE across all dimensions.
    """

    actual_values, reference_values = get_flat_output_and_desired(df)

    # Compute the squared differences
    squared_errors = (actual_values - reference_values) ** 2
    
    # Compute the mean of the sum of squared differences across all dimensions
    mse = np.mean(np.sum(squared_errors, axis=1))
    
    # Return the square root of the mean squared error (overall RMSE)
    overall_rmse = np.sqrt(mse)
    
    return overall_rmse