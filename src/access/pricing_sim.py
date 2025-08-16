"""
Price and Access Simulator for pharmaceutical market access modeling.

Maps pricing decisions to access tiers and adoption constraints.
Implements rule-based mapping with optional ML classifier overlay.

Access Tiers:
- OPEN: Broad coverage, minimal restrictions
- PA: Prior authorization required, moderate restrictions  
- NICHE: Highly restricted, specialty tier

Example usage:
    >>> tier = tier_from_price(1500)
    >>> print(f"Price $1500/month → {tier} tier")
    >>> ceiling = ACCESS_TIERS[tier]
    >>> print(f"Adoption ceiling: {ceiling:.0%}")
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings

# Import constants instead of hardcoding
try:
    from constants import (
        PRICE_THRESHOLDS, GTN_BY_TIER, ADOPTION_CEILING_BY_TIER
    )
except ImportError:
    # Fallback for backwards compatibility
    PRICE_THRESHOLDS = {'OPEN': 1000, 'PA': 2500, 'NICHE': float('inf')}
    GTN_BY_TIER = {'OPEN': 0.75, 'PA': 0.65, 'NICHE': 0.55}
    ADOPTION_CEILING_BY_TIER = {'OPEN': 1.0, 'PA': 0.6, 'NICHE': 0.25}

# Keep old name for backwards compatibility
ACCESS_TIERS = ADOPTION_CEILING_BY_TIER

def tier_from_price(list_price_month: float, 
                   thresholds: Optional[Dict[str, float]] = None) -> str:
    """
    Map monthly list price to access tier using rule-based thresholds.
    
    Args:
        list_price_month: Monthly list price in USD
        thresholds: Custom price thresholds (optional)
    
    Returns:
        str: Access tier ('OPEN', 'PA', or 'NICHE')
        
    Example:
        >>> tier_from_price(750)
        'OPEN'
        >>> tier_from_price(1800)
        'PA'
        >>> tier_from_price(3500)
        'NICHE'
    """
    if thresholds is None:
        # Use constants instead of hardcoded values
        if list_price_month < PRICE_THRESHOLDS['OPEN']:
            return "OPEN"
        elif list_price_month < PRICE_THRESHOLDS['PA']:
            return "PA"
        else:
            return "NICHE"
    else:
        # Custom thresholds provided
        if list_price_month < thresholds['open_max']:
            return "OPEN"
        elif list_price_month < thresholds['pa_max']:
            return "PA"
        else:
            return "NICHE"

def gtn_from_tier(access_tier: str, 
                 custom_gtn: Optional[Dict[str, float]] = None) -> float:
    """
    Get gross-to-net percentage for access tier.
    
    Args:
        access_tier: Access tier ('OPEN', 'PA', 'NICHE')
        custom_gtn: Custom GtN mapping (optional)
    
    Returns:
        float: Gross-to-net percentage
    """
    gtn_map = custom_gtn if custom_gtn else GTN_BY_TIER
    return gtn_map.get(access_tier, 0.65)  # Default to PA level

def adoption_ceiling_from_tier(access_tier: str,
                              custom_ceilings: Optional[Dict[str, float]] = None) -> float:
    """
    Get adoption ceiling (market accessibility) for access tier.
    
    Args:
        access_tier: Access tier ('OPEN', 'PA', 'NICHE')
        custom_ceilings: Custom ceiling mapping (optional)
    
    Returns:
        float: Adoption ceiling (0.0 to 1.0)
    """
    ceiling_map = custom_ceilings if custom_ceilings else ACCESS_TIERS
    return ceiling_map.get(access_tier, 0.6)  # Default to PA level

def apply_access_constraints(adopters: np.ndarray, 
                           access_tier: str,
                           custom_ceilings: Optional[Dict[str, float]] = None) -> np.ndarray:
    """
    Apply access tier constraints to adoption forecast.
    
    Args:
        adopters: Unconstrained adoption forecast
        access_tier: Access tier constraint
        custom_ceilings: Custom ceiling mapping (optional)
    
    Returns:
        np.ndarray: Constrained adoption forecast
        
    Example:
        >>> unconstrained = np.array([100, 200, 300, 250, 200])
        >>> constrained = apply_access_constraints(unconstrained, 'PA')
        >>> (constrained <= unconstrained).all()  # Always <= unconstrained
        True
    """
    ceiling = adoption_ceiling_from_tier(access_tier, custom_ceilings)
    
    # Apply ceiling to cumulative adoption
    cumulative = np.cumsum(adopters)
    market_size = cumulative[-1] if len(cumulative) > 0 else 0
    max_cumulative = market_size * ceiling
    
    # Constrain cumulative adoption
    constrained_cumulative = np.minimum(cumulative, max_cumulative)
    
    # Convert back to period-by-period adopters
    constrained_adopters = np.diff(np.concatenate([[0], constrained_cumulative]))
    
    return constrained_adopters

class AccessClassifier:
    """
    Optional ML classifier for predicting access favorability.
    
    This is a lightweight classifier that can be trained on payer policy data
    to predict whether a given price point will receive favorable access.
    """
    
    def __init__(self, random_state: int = 42):
        self.classifier = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=random_state
        )
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
    
    def fit(self, features: pd.DataFrame, access_tiers: List[str]) -> None:
        """
        Train classifier on payer policy data.
        
        Args:
            features: DataFrame with pricing/policy features
            access_tiers: List of access tier labels
        """
        # Encode access tiers
        encoded_tiers = self.label_encoder.fit_transform(access_tiers)
        
        # Fit classifier
        self.classifier.fit(features, encoded_tiers)
        self.is_fitted = True
    
    def predict_tier(self, features: pd.DataFrame) -> List[str]:
        """
        Predict access tier for new price points.
        
        Args:
            features: DataFrame with pricing/policy features
        
        Returns:
            List[str]: Predicted access tiers
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction")
        
        # Predict and decode
        encoded_pred = self.classifier.predict(features)
        return self.label_encoder.inverse_transform(encoded_pred).tolist()
    
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict access tier probabilities.
        
        Args:
            features: DataFrame with pricing/policy features
        
        Returns:
            np.ndarray: Probability matrix (n_samples x n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction")
        
        return self.classifier.predict_proba(features)

def create_pricing_scenario(price_range: Tuple[float, float],
                          n_points: int = 20,
                          custom_thresholds: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    Create pricing scenario analysis across price range.
    
    Args:
        price_range: (min_price, max_price) tuple
        n_points: Number of price points to analyze
        custom_thresholds: Custom access tier thresholds
    
    Returns:
        pd.DataFrame: Pricing scenario results
        
    Example:
        >>> scenario = create_pricing_scenario((500, 4000), n_points=10)
        >>> scenario.columns.tolist()
        ['price_monthly', 'access_tier', 'adoption_ceiling', 'gtn_pct', 'net_price']
    """
    min_price, max_price = price_range
    prices = np.linspace(min_price, max_price, n_points)
    
    results = []
    for price in prices:
        tier = tier_from_price(price, custom_thresholds)
        ceiling = adoption_ceiling_from_tier(tier)
        gtn = gtn_from_tier(tier)
        net_price = price * gtn
        
        results.append({
            'price_monthly': price,
            'access_tier': tier,
            'adoption_ceiling': ceiling,
            'gtn_pct': gtn,
            'net_price': net_price,
            'net_price_annual': net_price * 12
        })
    
    return pd.DataFrame(results)

def optimize_price_access_tradeoff(adopters: np.ndarray,
                                 price_range: Tuple[float, float],
                                 n_points: int = 50) -> Dict[str, any]:
    """
    Find optimal price point balancing access and revenue.
    
    Args:
        adopters: Base adoption forecast (unconstrained)
        price_range: (min_price, max_price) to evaluate
        n_points: Number of price points to test
    
    Returns:
        Dict with optimal price point and revenue analysis
    """
    scenario_df = create_pricing_scenario(price_range, n_points)
    
    results = []
    for _, row in scenario_df.iterrows():
        # Apply access constraints
        constrained_adopters = apply_access_constraints(adopters, row['access_tier'])
        total_patients = np.sum(constrained_adopters)
        
        # Calculate annual revenue (assuming persistence)
        annual_revenue = total_patients * row['net_price_annual']
        
        results.append({
            'price': row['price_monthly'],
            'tier': row['access_tier'],
            'patients': total_patients,
            'revenue': annual_revenue,
            'revenue_per_patient': row['net_price_annual']
        })
    
    results_df = pd.DataFrame(results)
    
    # Find optimal price (max revenue)
    optimal_idx = results_df['revenue'].idxmax()
    optimal_result = results_df.iloc[optimal_idx]
    
    return {
        'optimal_price': optimal_result['price'],
        'optimal_tier': optimal_result['tier'],
        'optimal_patients': optimal_result['patients'],
        'optimal_revenue': optimal_result['revenue'],
        'scenario_analysis': results_df
    }

if __name__ == "__main__":
    # Demo usage
    print("Price & Access Simulator Demo")
    print("=" * 40)
    
    # Test price-to-tier mapping
    test_prices = [750, 1500, 2800, 4000]
    for price in test_prices:
        tier = tier_from_price(price)
        ceiling = adoption_ceiling_from_tier(tier)
        gtn = gtn_from_tier(tier)
        print(f"${price:,}/month → {tier} tier (ceiling: {ceiling:.0%}, GtN: {gtn:.0%})")
    
    print("\nAccess constraint example:")
    # Example adoption curve
    base_adopters = np.array([1000, 2000, 3000, 2500, 2000, 1500, 1000])
    
    for tier in ['OPEN', 'PA', 'NICHE']:
        constrained = apply_access_constraints(base_adopters, tier)
        print(f"{tier}: {np.sum(constrained):,.0f} total patients "
              f"({100 * np.sum(constrained) / np.sum(base_adopters):.0f}% of unconstrained)")
    
    print("\nPricing scenario analysis:")
    scenario = create_pricing_scenario((500, 3500), n_points=8)
    print(scenario[['price_monthly', 'access_tier', 'adoption_ceiling', 'net_price']].round(0))
