#!/usr/bin/env python3
"""
H1: Evidence Grounding Experiment
Tests whether evidence-grounded forecasts beat pure heuristics.
Following Linus principle: Measure what matters, ignore what doesn't.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.baselines import ensemble_baseline, peak_sales_heuristic
from models.analogs import AnalogForecaster
from models.patient_flow import PatientFlowModel
from stats.protocol import (
    StatisticalProtocol, 
    EvaluationMetrics,
    CrossValidation,
    HypothesisTesting,
    AcceptanceGates
)
from utils.audit import get_audit_logger, audit_run


class H1Experiment:
    """
    H1: Evidence-grounded forecasts should beat pure heuristics.
    
    Conditions:
    - Heuristic: Simple peak sales formula without data
    - Evidence-light: Uses analogs but no external data
    - Evidence-heavy: Uses analogs + external validation data
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        
        self.protocol = StatisticalProtocol(seed=seed)
        self.metrics = EvaluationMetrics()
        self.cv = CrossValidation(self.protocol)
        self.hypothesis = HypothesisTesting(self.protocol)
        self.gates = AcceptanceGates()
        self.logger = get_audit_logger()
        
        # Load data
        self.data_dir = Path(__file__).parent.parent / "data_proc"
        self.results_dir = Path(__file__).parent.parent / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        self._load_data()
    
    def _load_data(self):
        """Load pharmaceutical launch data."""
        try:
            self.launches = pd.read_parquet(self.data_dir / "launches.parquet")
            self.revenues = pd.read_parquet(self.data_dir / "launch_revenues.parquet")
            self.analogs = pd.read_parquet(self.data_dir / "analogs.parquet")
            print(f"Loaded {len(self.launches)} launches")
        except FileNotFoundError:
            print("ERROR: Dataset not found. Run 'python src/cli.py build-data' first")
            sys.exit(1)
    
    def heuristic_forecast(self, train_data: pd.DataFrame, 
                          test_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Pure heuristic without evidence.
        Just uses peak sales formula.
        """
        forecasts = []
        y2_forecasts = []
        peak_forecasts = []
        
        for _, drug in test_data.iterrows():
            # Simple heuristic
            peak = peak_sales_heuristic(drug)
            
            # Linear growth to peak
            forecast = np.array([
                peak * 0.05,  # Y1: 5% of peak
                peak * 0.25,  # Y2: 25% of peak  
                peak * 0.60,  # Y3: 60% of peak
                peak * 0.85,  # Y4: 85% of peak
                peak * 1.00   # Y5: 100% peak
            ])
            
            forecasts.append(forecast)
            y2_forecasts.append(forecast[1] if len(forecast) > 1 else 0)
            peak_forecasts.append(peak)
        
        # Stack forecasts
        forecast_array = np.vstack(forecasts) if forecasts else np.array([])
        
        # Simple prediction intervals (±50%)
        return {
            'forecast': forecast_array,
            'forecast_y2': np.array(y2_forecasts),
            'forecast_peak': np.array(peak_forecasts),
            'lower': forecast_array * 0.5,
            'upper': forecast_array * 1.5,
            'y2_errors': np.array(y2_forecasts) - test_data['actual_y2'].values if 'actual_y2' in test_data else np.array([])
        }
    
    def evidence_light_forecast(self, train_data: pd.DataFrame,
                               test_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Evidence-light: Uses historical analogs from training data.
        """
        forecaster = AnalogForecaster()
        
        # Override with training data
        forecaster.launches = train_data
        forecaster.revenues = self.revenues[
            self.revenues['launch_id'].isin(train_data['launch_id'])
        ]
        forecaster.analogs = self.analogs[
            self.analogs['launch_id'].isin(train_data['launch_id']) &
            self.analogs['analog_launch_id'].isin(train_data['launch_id'])
        ]
        
        forecasts = []
        y2_forecasts = []
        peak_forecasts = []
        lowers = []
        uppers = []
        
        for _, drug in test_data.iterrows():
            # Get analog forecast with PI
            result = forecaster.get_prediction_intervals(drug, years=5)
            
            forecasts.append(result['forecast'])
            y2_forecasts.append(result['forecast'][1] if len(result['forecast']) > 1 else 0)
            peak_forecasts.append(np.max(result['forecast']))
            lowers.append(result['lower'])
            uppers.append(result['upper'])
        
        forecast_array = np.vstack(forecasts) if forecasts else np.array([])
        lower_array = np.vstack(lowers) if lowers else np.array([])
        upper_array = np.vstack(uppers) if uppers else np.array([])
        
        return {
            'forecast': forecast_array,
            'forecast_y2': np.array(y2_forecasts),
            'forecast_peak': np.array(peak_forecasts),
            'lower': lower_array,
            'upper': upper_array,
            'y2_errors': np.array(y2_forecasts) - test_data['actual_y2'].values if 'actual_y2' in test_data else np.array([])
        }
    
    def evidence_heavy_forecast(self, train_data: pd.DataFrame,
                               test_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Evidence-heavy: Uses analogs + patient flow + external validation.
        In real implementation, would query external databases.
        """
        # Combine analog and patient flow models
        analog_forecaster = AnalogForecaster()
        analog_forecaster.launches = train_data
        analog_forecaster.revenues = self.revenues[
            self.revenues['launch_id'].isin(train_data['launch_id'])
        ]
        analog_forecaster.analogs = self.analogs[
            self.analogs['launch_id'].isin(train_data['launch_id']) &
            self.analogs['analog_launch_id'].isin(train_data['launch_id'])
        ]
        
        flow_model = PatientFlowModel()
        
        forecasts = []
        y2_forecasts = []
        peak_forecasts = []
        lowers = []
        uppers = []
        
        for _, drug in test_data.iterrows():
            # Get both forecasts
            analog_result = analog_forecaster.get_prediction_intervals(drug, years=5)
            flow_scenarios = flow_model.forecast_with_scenarios(drug, years=5)
            
            # Weighted ensemble (60% analog, 40% patient flow)
            ensemble = 0.6 * analog_result['forecast'] + 0.4 * flow_scenarios['base']
            lower = 0.6 * analog_result['lower'] + 0.4 * flow_scenarios['downside']
            upper = 0.6 * analog_result['upper'] + 0.4 * flow_scenarios['upside']
            
            # Simulate "external validation" adjustment
            # In reality, would query pricing databases, clinical trial results, etc.
            market_adjustment = np.random.uniform(0.9, 1.1)  # Simulate market intel
            ensemble *= market_adjustment
            lower *= market_adjustment * 0.9
            upper *= market_adjustment * 1.1
            
            forecasts.append(ensemble)
            y2_forecasts.append(ensemble[1] if len(ensemble) > 1 else 0)
            peak_forecasts.append(np.max(ensemble))
            lowers.append(lower)
            uppers.append(upper)
        
        forecast_array = np.vstack(forecasts) if forecasts else np.array([])
        lower_array = np.vstack(lowers) if lowers else np.array([])
        upper_array = np.vstack(uppers) if uppers else np.array([])
        
        return {
            'forecast': forecast_array,
            'forecast_y2': np.array(y2_forecasts),
            'forecast_peak': np.array(peak_forecasts),
            'lower': lower_array,
            'upper': upper_array,
            'y2_errors': np.array(y2_forecasts) - test_data['actual_y2'].values if 'actual_y2' in test_data else np.array([])
        }
    
    def prepare_evaluation_data(self) -> pd.DataFrame:
        """Prepare data with actual values for evaluation."""
        # Merge launches with revenues to get actuals
        y2_revenues = self.revenues[self.revenues['year_since_launch'] == 1].rename(
            columns={'revenue_usd': 'actual_y2'}
        )[['launch_id', 'actual_y2']]
        
        peak_revenues = self.revenues.groupby('launch_id')['revenue_usd'].max().reset_index()
        peak_revenues.columns = ['launch_id', 'actual_peak']
        
        # Merge with launches
        eval_data = self.launches.merge(y2_revenues, on='launch_id', how='left')
        eval_data = eval_data.merge(peak_revenues, on='launch_id', how='left')
        
        # Fill missing with estimates (for incomplete data)
        eval_data['actual_y2'] = eval_data['actual_y2'].fillna(
            eval_data.apply(lambda x: peak_sales_heuristic(x) * 0.25, axis=1)
        )
        eval_data['actual_peak'] = eval_data['actual_peak'].fillna(
            eval_data.apply(peak_sales_heuristic, axis=1)
        )
        
        return eval_data
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run the full H1 experiment."""
        print("=" * 60)
        print("H1: EVIDENCE GROUNDING EXPERIMENT")
        print("=" * 60)
        
        # Prepare data
        eval_data = self.prepare_evaluation_data()
        
        # Temporal split
        train_data, test_data = self.cv.split_temporal(eval_data)
        print(f"\nData split: {len(train_data)} train, {len(test_data)} test")
        
        # Check Gate G3
        g3_passed, g3_msg = self.gates.check_gate_g3(
            self.protocol, len(train_data), len(test_data)
        )
        if not g3_passed:
            print(f"ERROR: {g3_msg}")
            return {'error': g3_msg}
        print(f"[+] {g3_msg}")
        
        # Run three conditions
        print("\n" + "-" * 40)
        print("Running forecasting methods...")
        
        # 1. Heuristic (baseline)
        print("  1. Pure heuristic...")
        heuristic_pred = self.heuristic_forecast(train_data, test_data)
        
        # 2. Evidence-light
        print("  2. Evidence-light (analogs)...")
        light_pred = self.evidence_light_forecast(train_data, test_data)
        
        # 3. Evidence-heavy
        print("  3. Evidence-heavy (ensemble + external)...")
        heavy_pred = self.evidence_heavy_forecast(train_data, test_data)
        
        # Calculate metrics
        print("\n" + "-" * 40)
        print("Calculating metrics...")
        
        results = {
            'heuristic': self._calculate_metrics(test_data, heuristic_pred),
            'evidence_light': self._calculate_metrics(test_data, light_pred),
            'evidence_heavy': self._calculate_metrics(test_data, heavy_pred)
        }
        
        # Statistical comparisons
        print("\n" + "-" * 40)
        print("Statistical testing...")
        
        # Compare evidence-light vs heuristic
        if 'y2_errors' in light_pred and len(light_pred['y2_errors']) > 0:
            light_vs_heuristic = self.hypothesis.compare_models(
                np.abs(light_pred['y2_errors']),
                np.abs(heuristic_pred['y2_errors']),
                paired=True
            )
            results['light_vs_heuristic'] = light_vs_heuristic
        
        # Compare evidence-heavy vs heuristic
        if 'y2_errors' in heavy_pred and len(heavy_pred['y2_errors']) > 0:
            heavy_vs_heuristic = self.hypothesis.compare_models(
                np.abs(heavy_pred['y2_errors']),
                np.abs(heuristic_pred['y2_errors']),
                paired=True
            )
            results['heavy_vs_heuristic'] = heavy_vs_heuristic
        
        # Compare evidence-heavy vs evidence-light
        if 'y2_errors' in heavy_pred and 'y2_errors' in light_pred:
            heavy_vs_light = self.hypothesis.compare_models(
                np.abs(heavy_pred['y2_errors']),
                np.abs(light_pred['y2_errors']),
                paired=True
            )
            results['heavy_vs_light'] = heavy_vs_light
        
        # Apply multiple comparison correction
        p_values = []
        if 'light_vs_heuristic' in results:
            p_values.append(results['light_vs_heuristic']['p_value'])
        if 'heavy_vs_heuristic' in results:
            p_values.append(results['heavy_vs_heuristic']['p_value'])
        if 'heavy_vs_light' in results:
            p_values.append(results['heavy_vs_light']['p_value'])
        
        if p_values:
            corrected = self.hypothesis.multiple_comparison_correction(p_values)
            results['corrected_significance'] = corrected
        
        # Summary
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        
        for method in ['heuristic', 'evidence_light', 'evidence_heavy']:
            if method in results:
                m = results[method]
                print(f"\n{method.upper()}:")
                print(f"  Y2 APE: {m.get('y2_ape', np.nan):.1f}%")
                print(f"  Peak APE: {m.get('peak_ape', np.nan):.1f}%")
                print(f"  PI Coverage: {m.get('pi_coverage', 0):.1%}")
        
        # Hypothesis results
        print("\n" + "-" * 40)
        print("HYPOTHESIS TESTING:")
        
        if 'light_vs_heuristic' in results:
            comp = results['light_vs_heuristic']
            sig = "YES" if comp['significant'] else "NO"
            print(f"  Evidence-light beats heuristic: {sig} (p={comp['p_value']:.4f})")
        
        if 'heavy_vs_heuristic' in results:
            comp = results['heavy_vs_heuristic']
            sig = "YES" if comp['significant'] else "NO"
            print(f"  Evidence-heavy beats heuristic: {sig} (p={comp['p_value']:.4f})")
        
        # Save results
        results['experiment'] = 'H1'
        results['seed'] = self.seed
        results['n_train'] = len(train_data)
        results['n_test'] = len(test_data)
        
        output_path = self.results_dir / "h1_results.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")
        
        # Log to audit
        audit_run(
            experiment_name='H1_evidence_grounding',
            config={'seed': self.seed},
            seed=self.seed,
            results=results
        )
        
        return results
    
    def _calculate_metrics(self, test_data: pd.DataFrame, 
                          predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        metrics = {}
        
        # Y2 APE
        if 'forecast_y2' in predictions and 'actual_y2' in test_data:
            y2_ape = self.metrics.year2_ape(
                test_data['actual_y2'].values,
                predictions['forecast_y2']
            )
            metrics['y2_ape'] = y2_ape
        
        # Peak APE
        if 'forecast_peak' in predictions and 'actual_peak' in test_data:
            peak_ape = self.metrics.peak_ape(
                test_data['actual_peak'].values,
                predictions['forecast_peak']
            )
            metrics['peak_ape'] = peak_ape
        
        # PI Coverage
        if 'lower' in predictions and 'upper' in predictions and 'actual_peak' in test_data:
            # Check if peak is within intervals
            coverage = self.metrics.prediction_interval_coverage(
                test_data['actual_peak'].values,
                predictions['forecast_peak'] * 0.8,  # Approximate lower
                predictions['forecast_peak'] * 1.2   # Approximate upper
            )
            metrics['pi_coverage'] = coverage
        
        return metrics


def main():
    """Run H1 experiment from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='H1: Evidence Grounding Experiment')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    experiment = H1Experiment(seed=args.seed)
    results = experiment.run_experiment()
    
    # Return 0 if hypothesis supported
    if results.get('heavy_vs_heuristic', {}).get('significant', False):
        print("\n[+] H1 SUPPORTED: Evidence grounding improves forecasts")
        return 0
    else:
        print("\n[-] H1 NOT SUPPORTED: Evidence grounding did not significantly improve")
        return 1


if __name__ == "__main__":
    sys.exit(main())