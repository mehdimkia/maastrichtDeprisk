# test_setup.py - Quick verification that everything is working
"""
Run this first to test if your setup is working correctly
"""

import sys
from pathlib import Path


def test_imports():
    """Test if all required packages are installed"""
    print("🧪 Testing imports...")

    try:
        import pandas as pd
        print("✅ pandas")

        import numpy as np
        print("✅ numpy")

        import sklearn
        print("✅ scikit-learn")

        import xgboost
        print("✅ xgboost")

        import optuna
        print("✅ optuna")

        from imblearn.over_sampling import SMOTE
        print("✅ imbalanced-learn")

        import joblib
        print("✅ joblib")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

    return True


def test_data_files():
    """Test if data files exist"""
    print("\n📁 Testing data files...")

    # Update this path to match your setup
    BASE = Path("/Volumes/education/DMS_744_S_Mirkialangaroodi/Databases")

    # If the above path doesn't work, try these alternatives:
    alternative_paths = [
        Path("/Users/mkia/Downloads/SummerPredectiveModel/maastrichtDeprisk/Databases"),
        Path("../Databases"),
        Path("./Databases")
    ]

    # Check if main path exists, if not try alternatives
    if not BASE.exists():
        print(f"❌ Main path not found: {BASE}")
        for alt_path in alternative_paths:
            if alt_path.exists():
                print(f"✅ Found alternative path: {alt_path}")
                BASE = alt_path
                break
        else:
            print("❌ No valid data path found. Please update BASE variable in test_setup.py")
            return False

    files_to_check = [
        "Week7_train_interactions.parquet",
        "Week7_test_interactions.parquet"
    ]

    all_found = True
    for file in files_to_check:
        file_path = BASE / file
        if file_path.exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} (not found at {file_path})")
            all_found = False

    return all_found


def test_imbalance_utils():
    """Test if our imbalance_utils module works"""
    print("\n🔧 Testing imbalance_utils...")

    try:
        from imbalance_utils import enhanced_sampling_strategies, get_cost_sensitive_weights
        print("✅ imbalance_utils imports work")

        # Test with dummy data
        import numpy as np
        X_dummy = np.random.rand(100, 5)
        y_dummy = np.array([0] * 85 + [1] * 15)  # Imbalanced like your data

        strategies = enhanced_sampling_strategies(X_dummy, y_dummy)
        print(f"✅ Found {len(strategies)} sampling strategies")

        weights = get_cost_sensitive_weights(y_dummy)
        print(f"✅ Class weights calculated: {weights}")

    except Exception as e:
        print(f"❌ imbalance_utils error: {e}")
        return False

    return True


def test_basic_preprocessing():
    """Test basic preprocessing with your data"""
    print("\n🔧 Testing preprocessing...")

    try:
        # Update this path to match your setup
        BASE = Path("/Volumes/education/DMS_744_S_Mirkialangaroodi/Databases")

        # If the above path doesn't work, try these alternatives:
        if not BASE.exists():
            alternative_paths = [
                Path("/Users/mkia/Downloads/SummerPredectiveModel/maastrichtDeprisk/Databases"),
                Path("../Databases"),
                Path("./Databases")
            ]

            for alt_path in alternative_paths:
                if alt_path.exists():
                    BASE = alt_path
                    break
            else:
                print("❌ Cannot find Databases directory")
                return False

        TRAIN_PQ = BASE / "Week7_train_interactions.parquet"

        if not TRAIN_PQ.exists():
            print(f"❌ Cannot find training data at {TRAIN_PQ}")
            return False

        import pandas as pd
        from sklearn.preprocessing import OneHotEncoder, SplineTransformer
        from sklearn.compose import ColumnTransformer
        import numpy as np

        # Load small sample
        train_df = pd.read_parquet(TRAIN_PQ)
        print(f"✅ Loaded training data: {train_df.shape}")

        y_train = train_df.pop("LD_PHQ9depr_event")
        print(f"✅ Target variable: {len(y_train)} samples")

        # Test preprocessing
        cat_cols = train_df.select_dtypes("category").columns.tolist()
        print(f"✅ Found {len(cat_cols)} categorical columns")

        spline = SplineTransformer(
            degree=3,
            knots=np.array([[360], [420], [480], [540], [600]]),
            include_bias=False,
        )

        preprocessor = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
            ("sleep_spline", spline, ["sleep_minutes"]),
        ], remainder="passthrough", sparse_threshold=0.3)

        # Test on small sample
        X_sample = train_df.head(100)
        X_processed = preprocessor.fit_transform(X_sample)
        print(f"✅ Preprocessing works: {X_processed.shape}")

        # Test feature names
        cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
        n_spline_features = preprocessor.named_transformers_['sleep_spline'].n_features_out_
        print(f"✅ Feature names work: {len(cat_feature_names)} cat + {n_spline_features} spline features")

    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        return False

    return True


def main():
    """Run all tests"""
    print("🧪 TESTING SETUP FOR IMPROVED TRAINING")
    print("=" * 50)

    tests = [
        ("Package Imports", test_imports),
        ("Data Files", test_data_files),
        ("Imbalance Utils", test_imbalance_utils),
        ("Basic Preprocessing", test_basic_preprocessing)
    ]

    all_passed = True
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            passed = test_func()
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ You can now run train_xgb_improved.py")
        print("\nNext steps:")
        print("1. cd to your src directory")
        print("2. python train_xgb_improved.py")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please fix the issues above before running the training script")

    return all_passed


if __name__ == "__main__":
    main()