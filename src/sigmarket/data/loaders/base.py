"""Base class for data loaders."""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, Dict, Any


class BaseDataLoader(ABC):
    """Abstract base class for all data loaders.

    Data loaders are responsible for loading raw data from various sources
    (Yahoo Finance, CSV files, synthetic generators, etc.) and returning
    a standardized pandas DataFrame format.

    All data loaders must implement:
        - load: Load data and return DataFrame
        - validate: Validate loaded data

    The returned DataFrame should have:
        - DatetimeIndex (for time series data)
        - Columns with appropriate names ('Close', 'Open', 'High', 'Low', 'Volume')
        - Clean data (no NaNs, no inf values)
    """

    def __init__(self, **kwargs):
        """Initialize data loader.

        Args:
            **kwargs: Loader-specific configuration
        """
        self.config = kwargs

    @abstractmethod
    def load(
        self,
        ticker: str,
        start: str,
        end: str,
        freq: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Load raw data and return DataFrame.

        Args:
            ticker: Ticker symbol or identifier
            start: Start date (YYYY-MM-DD format)
            end: End date (YYYY-MM-DD format)
            freq: Optional frequency for resampling ('D', 'W', 'M', 'Y')
            **kwargs: Additional loader-specific parameters

        Returns:
            DataFrame with DatetimeIndex and price/volume columns

        Raises:
            ValueError: If data cannot be loaded or is invalid
        """
        pass

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> bool:
        """Validate loaded DataFrame.

        Checks for:
            - DatetimeIndex
            - Required columns
            - No NaN values
            - No infinite values
            - Monotonic index

        Args:
            df: DataFrame to validate

        Returns:
            True if valid, False otherwise

        Raises:
            ValueError: If validation fails with specific error message
        """
        pass

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with standardized column names
        """
        # Common column name variations
        column_mapping = {
            'close': 'Close',
            'adj close': 'Close',
            'adjusted close': 'Close',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'volume': 'Volume',
        }

        # Create mapping for existing columns
        rename_dict = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower in column_mapping:
                rename_dict[col] = column_mapping[col_lower]

        if rename_dict:
            df = df.rename(columns=rename_dict)

        return df

    def _apply_frequency(
        self,
        df: pd.DataFrame,
        freq: str
    ) -> pd.DataFrame:
        """Resample DataFrame to specified frequency.

        Args:
            df: Input DataFrame
            freq: Frequency string ('D', 'W', 'M', 'Y')

        Returns:
            Resampled DataFrame
        """
        # Mapping to pandas frequency strings
        freq_map = {
            'D': 'D',      # Daily
            'W': 'W-FRI',  # Weekly (Friday)
            'M': 'M',      # Monthly (end of month)
            'Y': 'Y',      # Yearly
        }

        pandas_freq = freq_map.get(freq.upper(), freq)

        # Resample using appropriate aggregation
        resampled = df.resample(pandas_freq).agg({
            'Close': 'last',
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Volume': 'sum'
        })

        # Drop NaN rows (e.g., from incomplete periods)
        resampled = resampled.dropna()

        return resampled

    def _check_required_columns(
        self,
        df: pd.DataFrame,
        required_columns: list = None
    ) -> None:
        """Check if DataFrame has required columns.

        Args:
            df: DataFrame to check
            required_columns: List of required column names

        Raises:
            ValueError: If required columns are missing
        """
        if required_columns is None:
            required_columns = ['Close']

        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}. "
                f"Available columns: {list(df.columns)}"
            )

    def _check_datetime_index(self, df: pd.DataFrame) -> None:
        """Check if DataFrame has DatetimeIndex.

        Args:
            df: DataFrame to check

        Raises:
            ValueError: If index is not DatetimeIndex
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                f"DataFrame must have DatetimeIndex, got {type(df.index).__name__}"
            )

    def _check_no_nans(self, df: pd.DataFrame) -> None:
        """Check for NaN values in DataFrame.

        Args:
            df: DataFrame to check

        Raises:
            ValueError: If NaN values are found
        """
        if df.isnull().values.any():
            nan_counts = df.isnull().sum()
            nan_columns = nan_counts[nan_counts > 0]
            raise ValueError(
                f"DataFrame contains NaN values:\n{nan_columns}"
            )

    def _check_no_infs(self, df: pd.DataFrame) -> None:
        """Check for infinite values in DataFrame.

        Args:
            df: DataFrame to check

        Raises:
            ValueError: If infinite values are found
        """
        numeric_cols = df.select_dtypes(include=['float64', 'float32']).columns

        for col in numeric_cols:
            if (df[col] == float('inf')).any() or (df[col] == float('-inf')).any():
                raise ValueError(f"Column '{col}' contains infinite values")

    def __repr__(self) -> str:
        """String representation of data loader."""
        return f"{self.__class__.__name__}(config={self.config})"
