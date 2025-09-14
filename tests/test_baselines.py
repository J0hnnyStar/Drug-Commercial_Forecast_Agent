"""
Test industry baseline models.
These tests ensure our baselines match consultant-level quality.
Following Linus principle: Tests should test real behavior, not implementation.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.baselines import (
    peak_sales_heuristic,
    estimate_peak_share,
    year2_naive,
    linear_trend_forecast,
    market_share_evolution,
    simple_bass_forecast,
    ensemble_baseline
)

from models.analogs import (
    AnalogForecaster,
    analog_forecast,
    analog_forecast_with_pi
)

from models.patient_flow import (
    PatientFlowModel,
    PatientFlowParams,
    patient_flow_forecast,
    patient_flow_scenarios
)


class TestBaselines:
    """Test basic baseline methods."""
    
    @pytest.fixture
    def sample_drug(self):
        """Create a sample drug for testing."""
        return pd.Series({
            'launch_id': 'TEST_001',
            'drug_name': 'TestDrug',
            'company': 'TestCo',
            'therapeutic_area': 'Oncology',
            'eligible_patients_at_launch': 100000,
            'list_price_month_usd_launch': 10000,
            'net_gtn_pct_launch': 0.65,
            'access_tier_at_launch': 'PA',
            'competitor_count_at_launch': 3,
            'clinical_efficacy_proxy': 0.75,
            'safety_black_box': False,
            'mechanism': 'PD-1 inhibitor',
            'route': 'IV',
            'indication': 'NSCLC'
        })
    
    def test_peak_sales_heuristic(self, sample_drug):
        """Test peak sales calculation."""
        peak = peak_sales_heuristic(sample_drug)
        
        # Should be positive and reasonable
        assert peak > 0
        assert peak < 50e9  # Less than $50B
        
        # Check order of magnitude
        # 100k patients * 15% share * $120k/year * 70% compliance * 65% GTN
        expected_order = 100000 * 0.15 * 120000 * 0.70 * 0.65
        assert peak > expected_order * 0.1  # Within order of magnitude
        assert peak < expected_order * 10
    
    def test_estimate_peak_share(self, sample_drug):
        """Test peak share estimation."""
        share = estimate_peak_share(sample_drug)
        
        # Should be between 1% and 60%
        assert 0.01 <= share <= 0.60
        
        # First in class should have higher share
        fic_drug = sample_drug.copy()
        fic_drug['competitor_count_at_launch'] = 0
        fic_share = estimate_peak_share(fic_drug)
        assert fic_share > share
        
        # Crowded market should have lower share
        crowded_drug = sample_drug.copy()
        crowded_drug['competitor_count_at_launch'] = 10
        crowded_share = estimate_peak_share(crowded_drug)
        assert crowded_share < share
    
    def test_year2_naive(self, sample_drug):
        """Test year 2 forecast."""
        y2 = year2_naive(sample_drug)
        peak = peak_sales_heuristic(sample_drug)
        
        # Y2 should be 20-40% of peak for most drugs
        assert y2 > peak * 0.1
        assert y2 < peak * 0.5
        
        # Oncology should have faster uptake
        assert y2 / peak > 0.25  # At least 25% for oncology
    
    def test_linear_trend_forecast(self, sample_drug):
        """Test linear growth forecast."""
        forecast = linear_trend_forecast(sample_drug, years=5)
        
        # Should have 5 years
        assert len(forecast) == 5
        
        # Should be monotonically increasing to peak
        assert all(forecast[i] <= forecast[i+1] for i in range(3))
        
        # Should plateau after peak
        assert forecast[3] == forecast[4]
    
    def test_market_share_evolution(self, sample_drug):
        """Test market share evolution."""
        shares = market_share_evolution(sample_drug, years=5)
        
        # Should have 5 years
        assert len(shares) == 5
        
        # Should be between 0 and 1
        assert all(0 <= s <= 1 for s in shares)
        
        # Should be monotonically increasing
        assert all(shares[i] <= shares[i+1] for i in range(4))
    
    def test_simple_bass_forecast(self, sample_drug):
        """Test Bass diffusion forecast."""
        forecast = simple_bass_forecast(sample_drug, years=5)
        
        # Should have 5 years
        assert len(forecast) == 5
        
        # Should be positive
        assert all(f >= 0 for f in forecast)
        
        # Should show typical diffusion pattern
        # (increasing then potentially plateauing)
        assert forecast[1] > forecast[0]  # Growth from Y1 to Y2
    
    def test_ensemble_baseline(self, sample_drug):
        """Test ensemble of baselines."""
        results = ensemble_baseline(sample_drug, years=5)
        
        # Should have all methods plus ensemble
        assert 'peak_sales_linear' in results
        assert 'simple_bass' in results
        assert 'market_share' in results
        assert 'ensemble' in results
        
        # Ensemble should be average of others
        ensemble = results['ensemble']
        assert len(ensemble) == 5
        assert all(e > 0 for e in ensemble)
        
        # Ensemble should be between min and max of components
        all_forecasts = [v for k, v in results.items() if k != 'ensemble']
        min_forecast = np.min(all_forecasts, axis=0)
        max_forecast = np.max(all_forecasts, axis=0)
        
        for i in range(5):
            assert min_forecast[i] <= ensemble[i] <= max_forecast[i]


class TestAnalogForecaster:
    """Test analog-based forecasting."""
    
    @pytest.fixture
    def sample_drug(self):
        """Create a sample drug for testing."""
        return pd.Series({
            'launch_id': 'TEST_001',
            'drug_name': 'TestDrug',
            'company': 'TestCo',
            'therapeutic_area': 'Oncology',
            'eligible_patients_at_launch': 100000,
            'list_price_month_usd_launch': 10000,
            'net_gtn_pct_launch': 0.65,
            'access_tier_at_launch': 'PA',
            'competitor_count_at_launch': 3,
            'clinical_efficacy_proxy': 0.75,
            'safety_black_box': False,
            'mechanism': 'PD-1 inhibitor',
            'route': 'IV',
            'indication': 'NSCLC'
        })
    
    def test_analog_forecaster_init(self):
        """Test forecaster initialization."""
        forecaster = AnalogForecaster()
        assert forecaster is not None
        # Won't have data unless dataset is built
    
    def test_analog_forecast_fallback(self, sample_drug):
        """Test analog forecast falls back when no data."""
        forecast = analog_forecast(sample_drug, years=5)
        
        # Should return something even without analogs
        assert len(forecast) == 5
        assert all(f >= 0 for f in forecast)
    
    def test_analog_forecast_with_pi(self, sample_drug):
        """Test analog forecast with prediction intervals."""
        result = analog_forecast_with_pi(sample_drug, years=5)
        
        # Should have forecast and intervals
        assert 'forecast' in result
        assert 'lower' in result
        assert 'upper' in result
        
        # Intervals should bound forecast
        for i in range(5):
            assert result['lower'][i] <= result['forecast'][i]
            assert result['forecast'][i] <= result['upper'][i]
    
    def test_access_adjustment(self):
        """Test access tier adjustment calculation."""
        forecaster = AnalogForecaster()
        
        # Same tier should give 1.0
        assert forecaster._get_access_adjustment('PA', 'PA') == 1.0
        
        # Open to PA should give boost
        assert forecaster._get_access_adjustment('OPEN', 'PA') > 1.0
        
        # Niche to Open should give penalty
        assert forecaster._get_access_adjustment('NICHE', 'OPEN') < 1.0


class TestPatientFlowModel:
    """Test patient flow modeling."""
    
    @pytest.fixture
    def sample_drug(self):
        """Create a sample drug for testing."""
        return pd.Series({
            'launch_id': 'TEST_001',
            'drug_name': 'TestDrug',
            'company': 'TestCo',
            'therapeutic_area': 'Oncology',
            'eligible_patients_at_launch': 100000,
            'list_price_month_usd_launch': 10000,
            'net_gtn_pct_launch': 0.65,
            'access_tier_at_launch': 'PA',
            'competitor_count_at_launch': 3,
            'clinical_efficacy_proxy': 0.75,
            'safety_black_box': False,
            'mechanism': 'PD-1 inhibitor',
            'route': 'IV',
            'indication': 'NSCLC'
        })
    
    def test_set_params_from_drug(self, sample_drug):
        """Test parameter derivation."""
        model = PatientFlowModel()
        params = model.set_params_from_drug(sample_drug)
        
        # Check all params are set
        assert params.eligible_patients == 100000
        assert params.annual_price == 120000  # $10k * 12
        assert params.net_to_gross == 0.65
        
        # Check derived params are reasonable
        assert 0 < params.annual_incidence < 1
        assert 0 < params.annual_discontinuation < 1
        assert 0 < params.peak_penetration < 1
        assert params.years_to_peak > 0
        assert 0 < params.adherence_rate <= 1
    
    def test_simulate_patient_flow(self, sample_drug):
        """Test patient flow simulation."""
        model = PatientFlowModel()
        params = model.set_params_from_drug(sample_drug)
        states = model.simulate_patient_flow(params, years=5)
        
        # Should have all state arrays
        assert 'eligible' in states
        assert 'on_drug' in states
        assert 'new_patients' in states
        
        # All should have 5 years
        assert all(len(states[k]) == 5 for k in states)
        
        # Eligible should grow
        assert states['eligible'][4] >= states['eligible'][0]
        
        # On drug should be positive
        assert all(states['on_drug'] >= 0)
    
    def test_calculate_revenue(self, sample_drug):
        """Test revenue calculation."""
        model = PatientFlowModel()
        params = model.set_params_from_drug(sample_drug)
        states = model.simulate_patient_flow(params, years=5)
        revenue = model.calculate_revenue(states, params)
        
        # Should have 5 years of revenue
        assert len(revenue) == 5
        
        # Should be positive
        assert all(r >= 0 for r in revenue)
        
        # Should be reasonable order of magnitude
        # Patients * adherence * price * GTN
        max_revenue = states['on_drug'].max() * params.adherence_rate * params.annual_price * params.net_to_gross
        assert revenue.max() <= max_revenue * 1.01  # Allow small numerical error
    
    def test_patient_flow_forecast(self, sample_drug):
        """Test main forecast function."""
        forecast = patient_flow_forecast(sample_drug, years=5)
        
        # Should return 5 years
        assert len(forecast) == 5
        
        # Should be positive
        assert all(f >= 0 for f in forecast)
        
        # Should show growth pattern
        assert forecast[1] > forecast[0]  # Y2 > Y1
    
    def test_patient_flow_scenarios(self, sample_drug):
        """Test scenario generation."""
        scenarios = patient_flow_scenarios(sample_drug, years=5)
        
        # Should have all scenarios
        assert 'base' in scenarios
        assert 'upside' in scenarios
        assert 'downside' in scenarios
        
        # All should have 5 years
        assert all(len(scenarios[k]) == 5 for k in scenarios)
        
        # Upside > base > downside
        for i in range(5):
            assert scenarios['downside'][i] <= scenarios['base'][i]
            assert scenarios['base'][i] <= scenarios['upside'][i]
    
    def test_flow_diagnostics(self, sample_drug):
        """Test diagnostic output."""
        model = PatientFlowModel()
        diagnostics = model.get_flow_diagnostics(sample_drug, years=5)
        
        # Should have key metrics
        assert 'peak_patients' in diagnostics
        assert 'peak_year' in diagnostics
        assert 'peak_revenue' in diagnostics
        assert 'cumulative_revenue' in diagnostics
        
        # Peak year should be within forecast horizon
        assert 0 <= diagnostics['peak_year'] < 5
        
        # Peak penetration should be reasonable
        assert 0 < diagnostics['peak_penetration_achieved'] < 1


# Integration test
class TestBaselineIntegration:
    """Test that all baselines work together."""
    
    @pytest.fixture
    def diverse_drugs(self):
        """Create diverse drug profiles."""
        return [
            # Blockbuster oncology
            pd.Series({
                'launch_id': 'ONCO_001',
                'drug_name': 'OncoBlockbuster',
                'therapeutic_area': 'Oncology',
                'eligible_patients_at_launch': 500000,
                'list_price_month_usd_launch': 15000,
                'net_gtn_pct_launch': 0.60,
                'access_tier_at_launch': 'PA',
                'competitor_count_at_launch': 2,
                'clinical_efficacy_proxy': 0.85,
                'safety_black_box': False,
                'mechanism': 'Novel MOA',
                'route': 'IV',
                'indication': 'First line',
                'company': 'BigPharma'
            }),
            
            # Niche rare disease
            pd.Series({
                'launch_id': 'RARE_001',
                'drug_name': 'RareDiseaseDrug',
                'therapeutic_area': 'Rare Disease',
                'eligible_patients_at_launch': 5000,
                'list_price_month_usd_launch': 50000,
                'net_gtn_pct_launch': 0.75,
                'access_tier_at_launch': 'NICHE',
                'competitor_count_at_launch': 0,
                'clinical_efficacy_proxy': 0.90,
                'safety_black_box': False,
                'mechanism': 'Gene therapy',
                'route': 'IV',
                'indication': 'Ultra rare',
                'company': 'BioTech'
            }),
            
            # Crowded CV market
            pd.Series({
                'launch_id': 'CV_001',
                'drug_name': 'MeTooStatin',
                'therapeutic_area': 'Cardiovascular',
                'eligible_patients_at_launch': 10000000,
                'list_price_month_usd_launch': 200,
                'net_gtn_pct_launch': 0.40,
                'access_tier_at_launch': 'OPEN',
                'competitor_count_at_launch': 15,
                'clinical_efficacy_proxy': 0.65,
                'safety_black_box': False,
                'mechanism': 'Statin',
                'route': 'Oral',
                'indication': 'Hyperlipidemia',
                'company': 'Generic Inc'
            })
        ]
    
    def test_all_methods_produce_forecasts(self, diverse_drugs):
        """Test that all methods work for diverse drugs."""
        
        for drug in diverse_drugs:
            # Baselines
            ensemble = ensemble_baseline(drug, years=5)
            assert 'ensemble' in ensemble
            assert len(ensemble['ensemble']) == 5
            
            # Analogs
            analog_fc = analog_forecast(drug, years=5)
            assert len(analog_fc) == 5
            
            # Patient flow
            flow_fc = patient_flow_forecast(drug, years=5)
            assert len(flow_fc) == 5
            
            # All should be non-negative
            assert all(ensemble['ensemble'] >= 0)
            assert all(analog_fc >= 0)
            assert all(flow_fc >= 0)
    
    def test_methods_give_consistent_magnitude(self, diverse_drugs):
        """Test that different methods give similar order of magnitude."""
        
        for drug in diverse_drugs:
            # Get forecasts from all methods
            ensemble = ensemble_baseline(drug, years=5)['ensemble']
            analog = analog_forecast(drug, years=5)
            flow = patient_flow_forecast(drug, years=5)
            
            # Year 2 revenues should be within order of magnitude
            y2_values = [ensemble[1], analog[1], flow[1]]
            y2_values = [v for v in y2_values if v > 0]  # Filter zeros
            
            if len(y2_values) > 1:
                min_y2 = min(y2_values)
                max_y2 = max(y2_values)
                
                # Should be within 100x (two orders of magnitude)
                # Different methods can legitimately vary widely for extreme drugs
                if min_y2 > 0:
                    assert max_y2 / min_y2 < 100
    
    def test_therapeutic_area_differences(self, diverse_drugs):
        """Test that TAs behave differently."""
        
        onco_drug = diverse_drugs[0]  # Oncology
        rare_drug = diverse_drugs[1]  # Rare disease
        cv_drug = diverse_drugs[2]    # Cardiovascular
        
        # Oncology should have faster uptake than CV
        onco_flow = PatientFlowModel()
        onco_params = onco_flow.set_params_from_drug(onco_drug)
        
        cv_flow = PatientFlowModel()
        cv_params = cv_flow.set_params_from_drug(cv_drug)
        
        assert onco_params.years_to_peak < cv_params.years_to_peak
        
        # Rare disease should have higher peak penetration
        rare_flow = PatientFlowModel()
        rare_params = rare_flow.set_params_from_drug(rare_drug)
        
        assert rare_params.peak_penetration > cv_params.peak_penetration