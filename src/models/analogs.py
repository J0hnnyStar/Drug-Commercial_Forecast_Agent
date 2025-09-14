"""
Analog-based forecasting using historical similar drugs.
This is what real consultants use - find similar historical launches.
Following Linus principle: Real data beats theory every time.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class AnalogForecaster:
    """
    Forecast based on historical analogs.
    Industry standard approach used by LEK, ZS, IQVIA.
    """
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path(__file__).parent.parent.parent / "data_proc"
        self.launches = None
        self.revenues = None
        self.analogs = None
        self._load_data()
    
    def _load_data(self):
        """Load processed datasets."""
        try:
            self.launches = pd.read_parquet(self.data_dir / "launches.parquet")
            self.revenues = pd.read_parquet(self.data_dir / "launch_revenues.parquet")
            self.analogs = pd.read_parquet(self.data_dir / "analogs.parquet")
        except FileNotFoundError:
            # Allow initialization without data for testing
            pass
    
    def find_best_analogs(self, drug_row: pd.Series, n_analogs: int = 5) -> List[str]:
        """
        Find the best historical analogs for a drug.
        
        Args:
            drug_row: Row from launches.parquet
            n_analogs: Number of analogs to use
        
        Returns:
            List of launch_ids for best analogs
        """
        
        if self.analogs is None:
            return []
        
        # Get analogs for this drug
        drug_analogs = self.analogs[
            self.analogs['launch_id'] == drug_row['launch_id']
        ].sort_values('similarity_score', ascending=False)
        
        # Take top N
        best_analogs = drug_analogs.head(n_analogs)['analog_launch_id'].tolist()
        
        return best_analogs
    
    def get_analog_trajectories(self, analog_ids: List[str]) -> pd.DataFrame:
        """
        Get revenue trajectories for analog drugs.
        
        Args:
            analog_ids: List of launch_ids
        
        Returns:
            DataFrame with analog revenue trajectories
        """
        
        if self.revenues is None:
            return pd.DataFrame()
        
        # Get revenue data for analogs
        analog_revenues = self.revenues[
            self.revenues['launch_id'].isin(analog_ids)
        ]
        
        return analog_revenues
    
    def normalize_trajectories(self, drug_row: pd.Series, 
                              analog_revenues: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize analog trajectories to target drug characteristics.
        
        Args:
            drug_row: Target drug characteristics
            analog_revenues: Historical analog revenues
        
        Returns:
            Normalized revenue trajectories
        """
        
        if self.launches is None or analog_revenues.empty:
            return pd.DataFrame()
        
        normalized = []
        
        for analog_id in analog_revenues['launch_id'].unique():
            # Get analog characteristics
            analog_row = self.launches[
                self.launches['launch_id'] == analog_id
            ].iloc[0]
            
            # Get analog trajectory
            analog_traj = analog_revenues[
                analog_revenues['launch_id'] == analog_id
            ].copy()
            
            # Normalize for market size
            market_ratio = (drug_row['eligible_patients_at_launch'] / 
                          analog_row['eligible_patients_at_launch'])
            
            # Normalize for price
            price_ratio = ((drug_row['list_price_month_usd_launch'] * 12) /
                         (analog_row['list_price_month_usd_launch'] * 12))
            
            # Normalize for GTN
            gtn_ratio = (drug_row['net_gtn_pct_launch'] / 
                       analog_row['net_gtn_pct_launch'])
            
            # Apply normalization
            analog_traj['normalized_revenue'] = (
                analog_traj['revenue_usd'] * 
                market_ratio * price_ratio * gtn_ratio
            )
            
            # Adjust for access tier differences
            access_mult = self._get_access_adjustment(
                drug_row['access_tier_at_launch'],
                analog_row['access_tier_at_launch']
            )
            analog_traj['normalized_revenue'] *= access_mult
            
            # Adjust for efficacy differences
            efficacy_mult = (drug_row['clinical_efficacy_proxy'] / 
                           analog_row['clinical_efficacy_proxy'])
            analog_traj['normalized_revenue'] *= efficacy_mult
            
            normalized.append(analog_traj)
        
        if normalized:
            return pd.concat(normalized)
        return pd.DataFrame()
    
    def _get_access_adjustment(self, target_tier: str, analog_tier: str) -> float:
        """
        Get adjustment factor for access tier differences.
        
        Args:
            target_tier: Target drug access tier
            analog_tier: Analog drug access tier
        
        Returns:
            Adjustment multiplier
        """
        
        # Access tier impact on uptake
        tier_values = {
            'OPEN': 1.0,
            'PA': 0.6,
            'NICHE': 0.3
        }
        
        target_value = tier_values.get(target_tier, 0.6)
        analog_value = tier_values.get(analog_tier, 0.6)
        
        return target_value / analog_value
    
    def forecast_from_analogs(self, drug_row: pd.Series, 
                             years: int = 5,
                             n_analogs: int = 5,
                             method: str = 'weighted_mean') -> np.ndarray:
        """
        Generate forecast using analog method.
        
        Args:
            drug_row: Target drug to forecast
            years: Number of years to forecast
            n_analogs: Number of analogs to use
            method: Aggregation method ('mean', 'median', 'weighted_mean')
        
        Returns:
            Revenue forecast array
        """
        
        # Find best analogs
        analog_ids = self.find_best_analogs(drug_row, n_analogs)
        
        if not analog_ids:
            # No analogs found, use simple growth
            return self._fallback_forecast(drug_row, years)
        
        # Get analog trajectories
        analog_revenues = self.get_analog_trajectories(analog_ids)
        
        if analog_revenues.empty:
            return self._fallback_forecast(drug_row, years)
        
        # Normalize to target drug
        normalized = self.normalize_trajectories(drug_row, analog_revenues)
        
        # Aggregate by year
        forecast = np.zeros(years)
        
        for year in range(years):
            year_data = normalized[
                normalized['year_since_launch'] == year
            ]['normalized_revenue']
            
            if len(year_data) > 0:
                if method == 'mean':
                    forecast[year] = year_data.mean()
                elif method == 'median':
                    forecast[year] = year_data.median()
                elif method == 'weighted_mean':
                    # Weight by similarity score - properly align values with analog IDs
                    weights = []
                    values = []
                    
                    # Get normalized revenues for this year grouped by analog
                    year_normalized = normalized[
                        normalized['year_since_launch'] == year
                    ]
                    
                    for analog_id in analog_ids:
                        # Get revenue for this analog at this year
                        analog_revenue = year_normalized[
                            year_normalized['launch_id'] == analog_id
                        ]['normalized_revenue'].values
                        
                        if len(analog_revenue) > 0:
                            # Get similarity score for this analog
                            sim_score = self.analogs[
                                (self.analogs['launch_id'] == drug_row['launch_id']) &
                                (self.analogs['analog_launch_id'] == analog_id)
                            ]['similarity_score'].values
                            
                            if len(sim_score) > 0:
                                weights.append(sim_score[0])
                                values.append(analog_revenue[0])
                    
                    if weights:
                        forecast[year] = np.average(values, weights=weights)
                    else:
                        forecast[year] = year_data.mean()
            else:
                # Extrapolate if no data
                if year > 0 and forecast[year-1] > 0:
                    # Simple growth assumption
                    forecast[year] = forecast[year-1] * 1.15
        
        return forecast
    
    def _fallback_forecast(self, drug_row: pd.Series, years: int) -> np.ndarray:
        """Simple fallback when no analogs available."""
        
        # Import baseline method as fallback
        from .baselines import linear_trend_forecast
        return linear_trend_forecast(drug_row, years)
    
    def get_prediction_intervals(self, drug_row: pd.Series,
                                years: int = 5,
                                n_analogs: int = 10,
                                confidence: float = 0.8) -> Dict[str, np.ndarray]:
        """
        Generate prediction intervals using analog variability.
        
        Args:
            drug_row: Target drug
            years: Forecast horizon
            n_analogs: Number of analogs for interval estimation
            confidence: Confidence level (0.8 = 80% PI)
        
        Returns:
            Dict with 'forecast', 'lower', 'upper' arrays
        """
        
        # Get more analogs for interval estimation
        analog_ids = self.find_best_analogs(drug_row, n_analogs)
        
        if not analog_ids:
            # No intervals without analogs
            forecast = self._fallback_forecast(drug_row, years)
            return {
                'forecast': forecast,
                'lower': forecast * 0.7,
                'upper': forecast * 1.3
            }
        
        # Get all analog trajectories
        analog_revenues = self.get_analog_trajectories(analog_ids)
        normalized = self.normalize_trajectories(drug_row, analog_revenues)
        
        # Calculate percentiles by year
        forecast = np.zeros(years)
        lower = np.zeros(years)
        upper = np.zeros(years)
        
        alpha = (1 - confidence) / 2
        
        for year in range(years):
            year_data = normalized[
                normalized['year_since_launch'] == year
            ]['normalized_revenue'].values
            
            if len(year_data) > 2:
                forecast[year] = np.median(year_data)
                lower[year] = np.percentile(year_data, alpha * 100)
                upper[year] = np.percentile(year_data, (1-alpha) * 100)
            elif len(year_data) > 0:
                forecast[year] = np.mean(year_data)
                # Wide intervals with limited data
                lower[year] = forecast[year] * 0.5
                upper[year] = forecast[year] * 1.5
            else:
                # Extrapolate
                if year > 0:
                    growth = 1.15
                    forecast[year] = forecast[year-1] * growth
                    lower[year] = lower[year-1] * growth
                    upper[year] = upper[year-1] * growth
        
        return {
            'forecast': forecast,
            'lower': lower,
            'upper': upper
        }


def analog_forecast(drug_row: pd.Series, years: int = 5) -> np.ndarray:
    """
    Convenience function for analog forecasting.
    
    Args:
        drug_row: Drug to forecast
        years: Forecast horizon
    
    Returns:
        Revenue forecast
    """
    forecaster = AnalogForecaster()
    return forecaster.forecast_from_analogs(drug_row, years)


def analog_forecast_with_pi(drug_row: pd.Series, years: int = 5) -> Dict[str, np.ndarray]:
    """
    Analog forecast with prediction intervals.
    
    Args:
        drug_row: Drug to forecast
        years: Forecast horizon
    
    Returns:
        Dict with forecast and intervals
    """
    forecaster = AnalogForecaster()
    return forecaster.get_prediction_intervals(drug_row, years)