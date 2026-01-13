"""Data pipeline configuration."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from .base import BaseConfig


@dataclass
class DataPipelineConfig(BaseConfig):
    """Configuration for data pipeline.

    This configuration controls all aspects of data loading, preprocessing,
    signature computation, and dataset creation.

    Attributes:
        loader_type: Type of data loader ('yahoo', 'csv', 'parquet', 'rough_bergomi')
        ticker: Stock ticker symbol for single-asset mode
        tickers: List of ticker symbols for multi-asset mode
        start: Start date for data loading (YYYY-MM-DD format)
        end: End date for data loading (YYYY-MM-DD format)
        freq: Frequency for resampling ('D'=daily, 'W'=weekly, 'M'=monthly, 'Y'=yearly)
        sig_order: Order for signature computation (typically 3-4)
        use_logsig: Whether to use log-signatures (more compact)
        transforms: List of transformations to apply (['leadlag'], ['log_returns'], etc.)
        add_volume: Whether to include volume features
        add_volatility: Whether to compute and include volatility features
        add_time_features: Whether to add time-of-day/day-of-week encoding
        scaler_type: Type of scaler ('minmax', 'standard', 'robust')
        scale_range: Range for MinMaxScaler (min, max)
        multi_asset: Whether to handle multiple assets
        train_ratio: Ratio of data to use for training (rest for validation)
        cache_signatures: Whether to cache computed signatures to disk
        cache_dir: Directory for signature cache
    """

    # Data source
    loader_type: str = 'yahoo'
    ticker: str = 'AAPL'
    tickers: Optional[List[str]] = None
    start: str = '2010-01-01'
    end: str = '2020-01-01'
    freq: str = 'M'

    # Signature computation
    sig_order: int = 4
    use_logsig: bool = True

    # Transformations
    transforms: List[str] = field(default_factory=lambda: ['leadlag'])

    # Feature engineering
    add_volume: bool = False
    add_volatility: bool = False
    add_time_features: bool = False

    # Normalization
    scaler_type: str = 'minmax'
    scale_range: Tuple[float, float] = (0.00001, 0.99999)

    # Multi-asset
    multi_asset: bool = False

    # Data split
    train_ratio: float = 0.8

    # Caching
    cache_signatures: bool = True
    cache_dir: str = '~/.sigmarket/cache'

    def validate(self) -> None:
        """Validate configuration parameters."""
        # Signature order validation
        if self.sig_order < 2:
            raise ValueError(f"sig_order must be >= 2, got {self.sig_order}")

        # Train ratio validation
        if not 0 < self.train_ratio < 1:
            raise ValueError(f"train_ratio must be in (0, 1), got {self.train_ratio}")

        # Scaler type validation
        valid_scalers = ['minmax', 'standard', 'robust']
        if self.scaler_type not in valid_scalers:
            raise ValueError(f"scaler_type must be one of {valid_scalers}, got {self.scaler_type}")

        # Loader type validation
        valid_loaders = ['yahoo', 'csv', 'parquet', 'rough_bergomi']
        if self.loader_type not in valid_loaders:
            raise ValueError(f"loader_type must be one of {valid_loaders}, got {self.loader_type}")

        # Multi-asset validation
        if self.multi_asset and self.tickers is None:
            raise ValueError("tickers must be provided when multi_asset=True")

        # Scale range validation for minmax
        if self.scaler_type == 'minmax':
            if self.scale_range[0] >= self.scale_range[1]:
                raise ValueError(f"scale_range must have min < max, got {self.scale_range}")
