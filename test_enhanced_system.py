#!/usr/bin/env python3
"""
Test script for the enhanced Treasury yield analysis system.
This script tests the core functionality of the predictive modeling and enhanced data fetching.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test that all required modules can be imported"""
    print("Testing module imports...")

    try:
        from yield_predictor import YieldPredictor

        print("✓ YieldPredictor imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import YieldPredictor: {e}")
        return False

    try:
        from enhanced_data_fetcher import EnhancedDataFetcher

        print("✓ EnhancedDataFetcher imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import EnhancedDataFetcher: {e}")
        return False

    try:
        from yield_analyzer import YieldAnalyzer

        print("✓ YieldAnalyzer imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import YieldAnalyzer: {e}")
        return False

    return True


def test_predictor():
    """Test the YieldPredictor functionality"""
    print("\nTesting YieldPredictor...")

    try:
        predictor = YieldPredictor()
        print("✓ YieldPredictor initialized successfully")

        # Create synthetic data for testing
        dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")
        synthetic_data = pd.DataFrame(
            {
                "3M": np.random.normal(0.02, 0.005, len(dates)),
                "2Y": np.random.normal(0.025, 0.008, len(dates)),
                "5Y": np.random.normal(0.03, 0.01, len(dates)),
                "10Y": np.random.normal(0.035, 0.012, len(dates)),
                "30Y": np.random.normal(0.04, 0.015, len(dates)),
            },
            index=dates,
        )

        print("✓ Synthetic data created successfully")

        # Test VAR model fitting
        try:
            fitted_model, stationary_data = predictor.fit_var_model(synthetic_data)
            print("✓ VAR model fitted successfully")
            print(f"  - Model AIC: {fitted_model.aic:.2f}")
            print(f"  - Model BIC: {fitted_model.bic:.2f}")
            print(f"  - Optimal lags: {fitted_model.k_ar}")
        except Exception as e:
            print(f"✗ VAR model fitting failed: {e}")
            return False

        # Test forecast generation
        try:
            forecast_results = predictor.generate_forecast(
                synthetic_data, forecast_horizon=5, confidence_level=0.95
            )
            print("✓ Forecast generated successfully")
            print(f"  - Forecast horizon: {len(forecast_results['forecast'])} days")
            print(f"  - Tenors: {forecast_results['model_info']['tenors']}")
        except Exception as e:
            print(f"✗ Forecast generation failed: {e}")
            return False

        # Test forecast summary
        try:
            summary = predictor.get_forecast_summary(forecast_results)
            print("✓ Forecast summary generated successfully")
            print(f"  - Summary keys: {list(summary.keys())}")
        except Exception as e:
            print(f"✗ Forecast summary failed: {e}")
            return False

        return True

    except Exception as e:
        print(f"✗ YieldPredictor test failed: {e}")
        return False


def test_enhanced_fetcher():
    """Test the EnhancedDataFetcher functionality"""
    print("\nTesting EnhancedDataFetcher...")

    try:
        # Initialize without Redis for testing
        fetcher = EnhancedDataFetcher(redis_host=None)
        print("✓ EnhancedDataFetcher initialized successfully")

        # Test cache status
        cache_status = fetcher.get_cache_status()
        print(f"✓ Cache status: {cache_status['status']}")

        # Test data summary
        test_data = {
            "treasury": pd.DataFrame({"3M": [0.02, 0.021], "10Y": [0.035, 0.036]}),
            "macro": pd.DataFrame({"CPI": [100, 101], "GDP": [20000, 20100]}),
            "repo": pd.DataFrame({"GCF_REPO": [0.022, 0.023]}),
        }

        summary = fetcher.get_data_summary(test_data)
        print("✓ Data summary generated successfully")
        print(f"  - Sources: {list(summary.keys())}")

        # Test data merging
        merged_data = fetcher.merge_data_sources(test_data)
        print("✓ Data merging successful")
        print(f"  - Merged shape: {merged_data.shape}")

        return True

    except Exception as e:
        print(f"✗ EnhancedDataFetcher test failed: {e}")
        return False


def test_analyzer():
    """Test the YieldAnalyzer functionality"""
    print("\nTesting YieldAnalyzer...")

    try:
        analyzer = YieldAnalyzer()
        print("✓ YieldAnalyzer initialized successfully")

        # Create synthetic data
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
        synthetic_data = pd.DataFrame(
            {
                "3M": np.random.normal(0.02, 0.005, len(dates)),
                "2Y": np.random.normal(0.025, 0.008, len(dates)),
                "5Y": np.random.normal(0.03, 0.01, len(dates)),
                "10Y": np.random.normal(0.035, 0.012, len(dates)),
                "30Y": np.random.normal(0.04, 0.015, len(dates)),
            },
            index=dates,
        )

        # Test analysis
        analysis = analyzer.analyze_cycle(synthetic_data)
        print("✓ Cycle analysis completed successfully")
        print(f"  - Analysis keys: {list(analysis.keys())}")

        # Test PCA decomposition
        pca_results = analyzer.decompose_yields(synthetic_data)
        print("✓ PCA decomposition completed successfully")
        print(
            f"  - Explained variance: {[f'{v:.2%}' for v in pca_results['explained_variance'][:3]]}"
        )

        # Test regime detection
        regime_changes = analyzer.detect_regime_changes(synthetic_data)
        print("✓ Regime detection completed successfully")
        print(f"  - Detected regimes: {len(regime_changes)}")

        return True

    except Exception as e:
        print(f"✗ YieldAnalyzer test failed: {e}")
        return False


def test_integration():
    """Test integration between components"""
    print("\nTesting component integration...")

    try:
        from yield_predictor import YieldPredictor
        from enhanced_data_fetcher import EnhancedDataFetcher
        from yield_analyzer import YieldAnalyzer

        # Create synthetic data
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
        synthetic_data = pd.DataFrame(
            {
                "3M": np.random.normal(0.02, 0.005, len(dates)),
                "2Y": np.random.normal(0.025, 0.008, len(dates)),
                "5Y": np.random.normal(0.03, 0.01, len(dates)),
                "10Y": np.random.normal(0.035, 0.012, len(dates)),
                "30Y": np.random.normal(0.04, 0.015, len(dates)),
            },
            index=dates,
        )

        # Test full workflow
        predictor = YieldPredictor()
        analyzer = YieldAnalyzer()

        # 1. Analyze the data
        analysis = analyzer.analyze_cycle(synthetic_data)

        # 2. Generate forecast
        forecast_results = predictor.generate_forecast(
            synthetic_data, forecast_horizon=5
        )

        # 3. Get forecast summary
        summary = predictor.get_forecast_summary(forecast_results)

        print("✓ Integration test completed successfully")
        print(f"  - Analysis performed: {len(analysis) > 0}")
        print(f"  - Forecast generated: {forecast_results is not None}")
        print(f"  - Summary created: {summary is not None}")

        return True

    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("Enhanced Treasury Yield Analysis System - Test Suite")
    print("=" * 60)

    tests = [
        ("Module Imports", test_imports),
        ("YieldPredictor", test_predictor),
        ("EnhancedDataFetcher", test_enhanced_fetcher),
        ("YieldAnalyzer", test_analyzer),
        ("Integration", test_integration),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The enhanced system is ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
