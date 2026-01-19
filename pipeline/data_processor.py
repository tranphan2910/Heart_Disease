import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class DataProcessor:
    """
    Xử lý dữ liệu từ raw data đến data sẵn sàng cho modeling
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.numerical_cols = ['age', 'resting bp s', 'cholesterol', 'max heart rate', 'oldpeak']
        self.categorical_cols = ['sex', 'chest pain type', 'fasting blood sugar', 'resting ecg',
                                'exercise angina', 'ST slope']
        self.one_hot_cols = ['chest pain type', 'resting ecg', 'ST slope']
        
        # ⚠️ CHANGED: drop_cols will be determined dynamically via statistical testing
        # No longer hardcoded - will be calculated in identify_weak_features()
        self.drop_cols = []  # Will be populated during feature engineering
        self.weak_numerical_features = []  # Features with p-value >= 0.05
        self.weak_categorical_features = []  # Categories with <5% samples
        
        # Column name mapping for different dataset formats
        self.column_mapping = {
            # Resting blood pressure
            'trestbps': 'resting bp s',
            'resting_bp_s': 'resting bp s',
            'restingbp': 'resting bp s',
            'resting_bp': 'resting bp s',
            'rest_bp': 'resting bp s',
            'bp': 'resting bp s',
            
            # Cholesterol
            'chol': 'cholesterol',
            'serum_cholesterol': 'cholesterol',
            'cholestoral': 'cholesterol',
            
            # Max heart rate
            'thalach': 'max heart rate',
            'thalac': 'max heart rate',
            'max_heart_rate': 'max heart rate',
            'maxhr': 'max heart rate',
            'max_hr': 'max heart rate',
            'heart_rate': 'max heart rate',
            
            # Chest pain type
            'cp': 'chest pain type',
            'chest_pain_type': 'chest pain type',
            'chestpain': 'chest pain type',
            
            # Fasting blood sugar
            'fbs': 'fasting blood sugar',
            'fasting_blood_sugar': 'fasting blood sugar',
            'fastingbs': 'fasting blood sugar',
            
            # Resting ECG
            'restecg': 'resting ecg',
            'resting_ecg': 'resting ecg',
            'rest_ecg': 'resting ecg',
            
            # Exercise angina
            'exang': 'exercise angina',
            'exercise_angina': 'exercise angina',
            'exer_angina': 'exercise angina',
            'exerciseangina': 'exercise angina',
            
            # ST slope
            'slope': 'ST slope',
            'st_slope': 'ST slope',
            'stslope': 'ST slope',
            
            # Oldpeak
            'old_peak': 'oldpeak',
            'st_depression': 'oldpeak',
            
            # Target
            'num': 'target',
            'condition': 'target',
            'heart_disease': 'target',
            'disease': 'target',
            'output': 'target',
            'result': 'target',
            
            # Other
            'ca': 'ca',
            'thal': 'thal'
        }
    
    def normalize_column_names(self, data):
        """Normalize column names to standard format"""
        df = data.copy()
        
        # Convert all column names to lowercase first for matching
        df.columns = df.columns.str.lower().str.strip()
        
        # Apply mapping
        df = df.rename(columns=self.column_mapping)
        
        # Special case: normalize 'st slope' variations
        if 'st slope' in df.columns:
            df = df.rename(columns={'st slope': 'ST slope'})
        
        return df
    
    def load_data(self, file_path):
        """Load CSV data and normalize column names"""
        self.data = pd.read_csv(file_path)
        self.data = self.normalize_column_names(self.data)
        return self.data.copy()
    
    def validate_columns(self, data):
        """Validate that all required columns are present"""
        required_cols = set(self.numerical_cols + self.categorical_cols + ['target'])
        existing_cols = set(data.columns)
        missing_cols = required_cols - existing_cols
        
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}\n"
                f"Dataset columns: {list(data.columns)}\n"
                f"Please check your dataset format. Required columns are:\n"
                f"  Numerical: {self.numerical_cols}\n"
                f"  Categorical: {self.categorical_cols}\n"
                f"  Target: 'target'"
            )
        
        return True


    def drop_null_rows(self, data: pd.DataFrame):
        """Drop rows with any null (simple)"""
        before = data.shape[0]
        df = data.dropna()
        removed = before - df.shape[0]
        print(f"\n🧹 Dropped null rows: {removed} rows")
        return df, {"null_rows_removed": removed, "shape_after_nulls": df.shape}

    
    def handle_outliers(self, data):
        """Xử lý outliers theo phương pháp đã implement"""
        df = data.copy()
        
        # 1. Resting BP - Winsorize by IQR
        df = df[df['resting bp s'] > 0]
        Q1 = df['resting bp s'].quantile(0.25)
        Q3 = df['resting bp s'].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df['resting bp s'] = np.where(df['resting bp s'] < lower, lower, df['resting bp s'])
        df['resting bp s'] = np.where(df['resting bp s'] > upper, upper, df['resting bp s'])
        
        # 2. Cholesterol - Remove 0 values, log transform, winsorize
        df = df[df['cholesterol'] > 0]
        df['cholesterol'] = np.log1p(df['cholesterol'])
        df['cholesterol'] = winsorize(df['cholesterol'], limits=[0.02, 0.02])
        
        # 3. Oldpeak - Remove negative, log transform
        df = df[df['oldpeak'] >= 0]
        df['oldpeak'] = np.log1p(df['oldpeak'])
        
        # 4. Max Heart Rate - Winsorize 1% both sides
        df['max heart rate'] = winsorize(df['max heart rate'], limits=[0.01, 0.01])
        
        return df
    
    def identify_weak_features(self, data, target_col='target', p_threshold=0.05, sample_threshold=0.05):
        """
        Dynamically identify weak features based on statistical testing
        
        Args:
            data: DataFrame with features and target
            target_col: Name of target column
            p_threshold: P-value threshold for Welch's t-test (default 0.05)
            sample_threshold: Minimum sample percentage threshold (default 0.05 = 5%)
            
        Returns:
            weak_features: List of features to drop
        """
        df = data.copy()
        weak_features = []
        
        # Separate features and target
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
        print("\n" + "="*60)
        print("🔍 Phase 1: Statistical Testing - Identifying Weak Features")
        print("="*60)
        
        # 1. Test numerical features using Welch's t-test
        print("\n📊 Testing Numerical Features (Welch's t-test)...")
        for feature in self.numerical_cols:
            if feature in X.columns:
                group_0 = X[y == 0][feature]
                group_1 = X[y == 1][feature]
                
                # Welch's t-test (doesn't assume equal variances)
                t_stat, p_value = stats.ttest_ind(group_0, group_1, equal_var=False)
                
                if p_value >= p_threshold:
                    weak_features.append(feature)
                    self.weak_numerical_features.append(feature)
                    print(f"  ❌ {feature}: p-value={p_value:.4f} >= {p_threshold} (NOT significant) → REMOVE")
                else:
                    print(f"  ✅ {feature}: p-value={p_value:.4f} < {p_threshold} (significant) → KEEP")
        
        # 2. Test categorical features after one-hot encoding
        print("\n📊 Testing Categorical Features (Sample Distribution)...")
        
        # Create temporary df with one-hot encoding for analysis
        df_temp = pd.get_dummies(X, columns=self.one_hot_cols, dtype=int)
        total_samples = len(df_temp)
        
        # Get all one-hot encoded columns
        for original_col in self.one_hot_cols:
            one_hot_cols = [col for col in df_temp.columns if col.startswith(original_col + '_')]
            
            print(f"\n  Analyzing '{original_col}' (One-Hot Encoded):")
            for col in one_hot_cols:
                sample_count = df_temp[col].sum()
                sample_percentage = sample_count / total_samples
                
                if sample_percentage < sample_threshold:
                    weak_features.append(col)
                    self.weak_categorical_features.append(col)
                    print(f"    ❌ {col}: {sample_count} samples ({sample_percentage*100:.1f}%) < {sample_threshold*100}% → REMOVE")
                else:
                    print(f"    ✅ {col}: {sample_count} samples ({sample_percentage*100:.1f}%) → KEEP")
        
        # 3. Always remove AgeBand if exists (derived feature)
        if 'AgeBand' in X.columns:
            weak_features.append('AgeBand')
            print(f"\n  ❌ AgeBand: Derived feature → REMOVE")
        
        # Update drop_cols
        self.drop_cols = weak_features
        
        print("\n" + "="*60)
        print(f"✅ Identified {len(weak_features)} weak features to remove")
        print("="*60)
        print(f"Weak features: {weak_features}")
        print()
        
        return weak_features
    
    def feature_engineering(self, data, target_col='target', auto_identify_weak=True):
        """
        Feature engineering: One-hot encoding và drop weak features
        
        Args:
            data: DataFrame with features and target
            target_col: Name of target column
            auto_identify_weak: If True, automatically identify weak features via statistical testing
        """
        df = data.copy()
        
        # Automatically identify weak features if requested
        if auto_identify_weak and target_col in df.columns:
            print("🔍 Auto-identifying weak features via statistical testing...")
            self.identify_weak_features(df, target_col=target_col)
            
            # Remove target before feature engineering
            y = df[target_col]
            X = df.drop(columns=[target_col])
        else:
            # Manual mode - use predefined drop_cols
            if target_col in df.columns:
                y = df[target_col]
                X = df.drop(columns=[target_col])
            else:
                X = df
                y = None
        
        # One-hot encoding
        X = pd.get_dummies(X, columns=self.one_hot_cols, dtype=int)
        
        # Drop weak features (động học từ statistical testing)
        # Check xem columns có tồn tại không trước khi drop
        existing_drop_cols = [col for col in self.drop_cols if col in X.columns]
        if existing_drop_cols:
            print(f"\n🗑️ Dropping {len(existing_drop_cols)} weak features: {existing_drop_cols}")
            X = X.drop(existing_drop_cols, axis=1)
        
        # Recombine with target if it existed
        if y is not None:
            result = pd.concat([X, y], axis=1)
        else:
            result = X
        
        return result
    
    def get_outlier_stats(self, data):
        """Thống kê số lượng outliers"""
        Q1 = data[self.numerical_cols].quantile(0.25)
        Q3 = data[self.numerical_cols].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_count = ((data[self.numerical_cols] < lower_bound) | 
                         (data[self.numerical_cols] > upper_bound)).sum()
        return outliers_count
    
    def process_pipeline(self, file_path):
        """
        Pipeline hoàn chỉnh từ raw data đến processed data
        
        Returns:
            processed_data: DataFrame đã xử lý
            stats: Dictionary chứa các thống kê
        """
        stats = {}
        
        # 1. Load data
        raw_data = self.load_data(file_path)
        stats['original_shape'] = raw_data.shape
        stats['missing_values'] = raw_data.isnull().sum().to_dict()
        
        # 2. Validate columns
        self.validate_columns(raw_data)

        #3. Drop null rows
        raw_data, null_stats = self.drop_null_rows(raw_data)
        stats.update(null_stats)

        
        # 4. Outlier statistics before handling
        stats['outliers_before'] = self.get_outlier_stats(raw_data).to_dict()
        
        # 5. Handle outliers
        data_cleaned = self.handle_outliers(raw_data)
        stats['shape_after_outlier_handling'] = data_cleaned.shape
        stats['outliers_after'] = self.get_outlier_stats(data_cleaned).to_dict()

        # 6. Feature engineering
        processed_data = self.feature_engineering(data_cleaned)
        stats['final_shape'] = processed_data.shape
        stats['final_features'] = processed_data.columns.tolist()
        
        return processed_data, stats
    
    def get_X_y(self, data):
        """Tách features và target"""
        X = data.drop('target', axis=1)
        y = data['target']
        return X, y


if __name__ == "__main__":
    # Test the data processor
    processor = DataProcessor()
    data, stats = processor.process_pipeline('heart_statlog_cleveland_hungary_final.csv')
    print("Processing completed!")
    print(f"Final shape: {stats['final_shape']}")
    print(f"Number of features: {len(stats['final_features'])}")
