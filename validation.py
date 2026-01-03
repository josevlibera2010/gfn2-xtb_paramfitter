"""
Input validation utilities for the GFN2-xTB parameter fitter.

Provides validation functions with logging support for warnings
and exceptions for critical validation failures.
"""

import logging
import os
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def validate_positive(value: float, name: str) -> None:
    """Validate that a value is positive (> 0).

    Args:
        value: The value to check
        name: Name of the parameter (for error messages)

    Raises:
        ValueError: If value <= 0
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_non_zero(value: float, name: str) -> None:
    """Validate that a value is not zero.

    Args:
        value: The value to check
        name: Name of the parameter (for error messages)

    Raises:
        ValueError: If value == 0
    """
    if value == 0:
        raise ValueError(f"{name} cannot be zero")


def validate_non_empty_array(arr: Sequence, name: str) -> None:
    """Validate that an array/sequence is not empty.

    Args:
        arr: The array or sequence to check
        name: Name of the parameter (for error messages)

    Raises:
        ValueError: If array is empty
    """
    if len(arr) == 0:
        raise ValueError(f"{name} array is empty")


def validate_min_length(arr: Sequence, min_len: int, name: str) -> None:
    """Validate that an array has at least min_len elements.

    Args:
        arr: The array or sequence to check
        min_len: Minimum required length
        name: Name of the parameter (for error messages)

    Raises:
        ValueError: If len(arr) < min_len
    """
    if len(arr) < min_len:
        raise ValueError(f"{name} must have at least {min_len} elements, got {len(arr)}")


def validate_directory_exists(path: str, name: str) -> None:
    """Validate that a directory exists.

    Args:
        path: Path to the directory
        name: Name of the parameter (for error messages)

    Raises:
        ValueError: If directory does not exist
    """
    if not os.path.isdir(path):
        raise ValueError(f"{name} directory does not exist: {path}")


def validate_file_exists(path: str, name: str) -> None:
    """Validate that a file exists.

    Args:
        path: Path to the file
        name: Name of the parameter (for error messages)

    Raises:
        ValueError: If file does not exist
    """
    if not os.path.isfile(path):
        raise ValueError(f"{name} file does not exist: {path}")


def validate_env_var(var_name: str) -> str:
    """Validate that an environment variable is set and return its value.

    Args:
        var_name: Name of the environment variable

    Returns:
        The value of the environment variable

    Raises:
        EnvironmentError: If the variable is not set
    """
    value = os.environ.get(var_name)
    if value is None:
        raise EnvironmentError(
            f"Environment variable {var_name} is not set. "
            f"Please activate the appropriate conda environment."
        )
    return value


def validate_no_nan_inf(arr: np.ndarray, name: str) -> None:
    """Validate that an array contains no NaN or Inf values.

    Args:
        arr: The numpy array to check
        name: Name of the parameter (for error messages)

    Raises:
        ValueError: If array contains NaN or Inf
    """
    if not np.all(np.isfinite(arr)):
        nan_count = np.sum(np.isnan(arr))
        inf_count = np.sum(np.isinf(arr))
        raise ValueError(
            f"{name} contains invalid values: {nan_count} NaN, {inf_count} Inf"
        )