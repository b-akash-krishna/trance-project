"""
Standalone Prediction API for TRANCE
Save as: src/models/prediction_api.py

Usage:
    python src/models/prediction_api.py --age 65 --gender M --charlson 3 --prior_admits 2

Or use in Python:
    from src.models.prediction_api import predict_readmission
    risk = predict_readmission(age=65, gender='M', charlson_score=3)
"""

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from pathlib import Path
import json
import argparse
from typing import Dict, Optional

class ReadmissionPredictor:
    """
    Readmission prediction model wrapper
    """
    
    def __init__(self, model_dir: Path = None):
        """Initialize predictor with trained models"""
        if model_dir is None:
            model_dir = Path(__file__).parent.parent.parent / 'outputs/models'
        
        self.model_dir = Path(model_dir)
        
        # Load models
        self.model = lgb.Booster(model_file=str(self.model_dir / 'fused_model.txt'))
        self.calibrator = joblib.load(self.model_dir / 'calibrator.pkl')
        
        # Load feature info
        feature_info_path = self.model_dir.parent.parent / 'data/processed/feature_info.json'
        with open(feature_info_path, 'r') as f:
            self.feature_info = json.load(f)
        
        self.feature_names = self.feature_info['feature_names']
        
        print(f"✓ Loaded model from {self.model_dir}")
        print(f"✓ Model expects {len(self.feature_names)} features")
    
    def create_feature_vector(self, 
                            age: float,
                            gender: str,
                            ethnicity: str = 'WHITE',
                            insurance: str = 'Medicare',
                            charlson_score: int = 0,
                            prior_admissions_180d: int = 0,
                            los_days: float = 3.0,
                            dx_heart_failure: bool = False,
                            dx_copd: bool = False,
                            dx_diabetes: bool = False,
                            dx_renal_failure: bool = False,
                            had_icu_stay: bool = False,
                            discharge_weekend: bool = False,
                            n_medications: int = 5,
                            n_diagnoses: int = 3,
                            lab_creatinine: Optional[float] = None,
                            lab_hemoglobin: Optional[float] = None,
                            clinical_note: Optional[str] = None,
                            **kwargs) -> np.ndarray:
        """
        Create feature vector from patient data
        
        Args:
            age: Patient age
            gender: 'M' or 'F'
            ethnicity: Ethnicity category
            insurance: Insurance type
            charlson_score: Charlson comorbidity index
            prior_admissions_180d: Prior admissions in 6 months
            los_days: Length of stay in days
            dx_heart_failure: Heart failure diagnosis flag
            dx_copd: COPD diagnosis flag
            dx_diabetes: Diabetes diagnosis flag
            dx_renal_failure: Renal failure diagnosis flag
            had_icu_stay: ICU admission flag
            discharge_weekend: Weekend discharge flag
            n_medications: Number of medications
            n_diagnoses: Number of diagnoses
            lab_creatinine: Last creatinine value (mg/dL)
            lab_hemoglobin: Last hemoglobin value (g/dL)
            clinical_note: Discharge summary text (optional)
            **kwargs: Additional features
            
        Returns:
            Feature vector as numpy array
        """
        
        features_dict = {}
        
        # Demographics
        features_dict['age'] = age
        features_dict['gender_M'] = 1 if gender.upper() in ['M', 'MALE'] else 0
        features_dict['gender_F'] = 1 if gender.upper() in ['F', 'FEMALE'] else 0
        
        # Ethnicity dummies
        for eth in ['WHITE', 'BLACK', 'HISPANIC', 'ASIAN', 'OTHER', 'UNKNOWN']:
            features_dict[f'ethnicity_{eth}'] = 1 if ethnicity.upper() == eth else 0
        
        # Insurance dummies
        insurance_clean = insurance.replace(' ', '')
        for ins in ['Medicare', 'Medicaid', 'Private', 'SelfPay']:
            features_dict[f'insurance_{ins}'] = 1 if insurance_clean.lower() == ins.lower() else 0
        
        # Clinical history
        features_dict['charlson_score'] = charlson_score
        features_dict['prior_admissions_180d'] = prior_admissions_180d
        features_dict['prior_admissions_90d'] = min(prior_admissions_180d, prior_admissions_180d // 2)
        features_dict['prior_admissions_365d'] = prior_admissions_180d  # Approximate
        features_dict['frequent_flyer'] = 1 if prior_admissions_180d >= 3 else 0
        
        # Length of stay
        features_dict['los_days'] = los_days
        features_dict['los_hours'] = los_days * 24
        
        # LOS categories
        if los_days <= 2:
            features_dict['los_short'] = 1
            features_dict['los_medium'] = 0
            features_dict['los_long'] = 0
            features_dict['los_very_long'] = 0
        elif los_days <= 5:
            features_dict['los_short'] = 0
            features_dict['los_medium'] = 1
            features_dict['los_long'] = 0
            features_dict['los_very_long'] = 0
        elif los_days <= 10:
            features_dict['los_short'] = 0
            features_dict['los_medium'] = 0
            features_dict['los_long'] = 1
            features_dict['los_very_long'] = 0
        else:
            features_dict['los_short'] = 0
            features_dict['los_medium'] = 0
            features_dict['los_long'] = 0
            features_dict['los_very_long'] = 1
        
        # Diagnoses
        features_dict['dx_heart_failure'] = 1 if dx_heart_failure else 0
        features_dict['dx_copd'] = 1 if dx_copd else 0
        features_dict['dx_diabetes'] = 1 if dx_diabetes else 0
        features_dict['dx_renal_failure'] = 1 if dx_renal_failure else 0
        features_dict['n_diagnoses'] = n_diagnoses
        
        # Hospital course
        features_dict['had_icu_stay'] = 1 if had_icu_stay else 0
        features_dict['n_icu_stays'] = 1 if had_icu_stay else 0
        features_dict['total_icu_days'] = 1.0 if had_icu_stay else 0.0
        features_dict['discharge_weekend'] = 1 if discharge_weekend else 0
        features_dict['n_medications'] = n_medications
        features_dict['n_procedures'] = 0  # Default
        
        # Labs
        if lab_creatinine is not None:
            features_dict['lab_creatinine_last'] = lab_creatinine
            features_dict['lab_creatinine_last_missing'] = 0
        else:
            features_dict['lab_creatinine_last'] = 1.0  # Default median
            features_dict['lab_creatinine_last_missing'] = 1
        
        if lab_hemoglobin is not None:
            features_dict['lab_hemoglobin_last'] = lab_hemoglobin
            features_dict['lab_hemoglobin_last_missing'] = 0
        else:
            features_dict['lab_hemoglobin_last'] = 12.0  # Default median
            features_dict['lab_hemoglobin_last_missing'] = 1
        
        # Process clinical note if provided
        if clinical_note:
            try:
                embeddings = self._generate_embeddings(clinical_note)
                for i, emb_val in enumerate(embeddings):
                    features_dict[f'emb_{i}'] = emb_val
                features_dict['has_discharge_note'] = 1
            except Exception as e:
                print(f"Warning: Could not process clinical note: {e}")
                for i in range(768):
                    features_dict[f'emb_{i}'] = 0
                features_dict['has_discharge_note'] = 0
        else:
            # No note - zero embeddings
            for i in range(768):
                features_dict[f'emb_{i}'] = 0
            features_dict['has_discharge_note'] = 0
        
        # Add any additional kwargs
        features_dict.update(kwargs)
        
        # Fill missing features with defaults
        for feat in self.feature_names:
            if feat not in features_dict:
                features_dict[feat] = 0
        
        # Create vector in correct order
        feature_vector = [features_dict.get(feat, 0) for feat in self.feature_names]
        
        return np.array(feature_vector)
    
    def _generate_embeddings(self, text: str) -> np.ndarray:
        """Generate embeddings from clinical text using ClinicalT5"""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            # Load model (cached)
            if not hasattr(self, '_embedding_model'):
                self._tokenizer = AutoTokenizer.from_pretrained("luqh/ClinicalT5-large")
                self._embedding_model = AutoModel.from_pretrained("luqh/ClinicalT5-large")
                self._embedding_model.eval()
            
            # Tokenize and generate embedding
            inputs = self._tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
            
            with torch.no_grad():
                outputs = self._embedding_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            
            return embeddings
            
        except ImportError:
            print("Warning: transformers library not available. Install with: pip install transformers torch")
            return np.zeros(768)
    
    def predict(self, **patient_data) -> Dict:
        """
        Predict readmission risk for a patient
        
        Args:
            **patient_data: Patient features (see create_feature_vector for details)
            
        Returns:
            Dictionary with prediction results
        """
        
        # Create feature vector
        X = self.create_feature_vector(**patient_data).reshape(1, -1)
        
        # Make prediction
        raw_prob = self.model.predict(X)[0]
        calibrated_prob = self.calibrator.predict(np.array([raw_prob]))[0]
        risk_score = calibrated_prob * 100
        
        # Determine risk level
        if risk_score >= 70:
            risk_level = "VERY HIGH RISK"
            risk_category = "critical"
        elif risk_score >= 50:
            risk_level = "HIGH RISK"
            risk_category = "high"
        elif risk_score >= 30:
            risk_level = "MODERATE RISK"
            risk_category = "moderate"
        else:
            risk_level = "LOW RISK"
            risk_category = "low"
        
        return {
            'risk_score': float(risk_score),
            'risk_probability': float(calibrated_prob),
            'risk_level': risk_level,
            'risk_category': risk_category,
            'raw_probability': float(raw_prob),
            'model_version': '1.0.0'
        }
    
    def predict_batch(self, patients_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict readmission risk for multiple patients
        
        Args:
            patients_df: DataFrame with patient data (columns matching feature names)
            
        Returns:
            DataFrame with predictions added
        """
        
        results = []
        
        for idx, row in patients_df.iterrows():
            patient_data = row.to_dict()
            result = self.predict(**patient_data)
            results.append(result)
        
        result_df = pd.DataFrame(results)
        return pd.concat([patients_df.reset_index(drop=True), result_df], axis=1)


def predict_readmission(**patient_data):
    """
    Convenience function for making predictions
    
    Example:
        >>> risk = predict_readmission(
        ...     age=65,
        ...     gender='M',
        ...     charlson_score=3,
        ...     prior_admissions_180d=2,
        ...     los_days=5.0
        ... )
        >>> print(f"Risk: {risk['risk_score']:.1f}%")
    """
    predictor = ReadmissionPredictor()
    return predictor.predict(**patient_data)


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description='TRANCE Readmission Prediction')
    
    # Required arguments
    parser.add_argument('--age', type=float, required=True, help='Patient age')
    parser.add_argument('--gender', type=str, required=True, choices=['M', 'F', 'Male', 'Female'], help='Gender')
    
    # Optional clinical arguments
    parser.add_argument('--ethnicity', type=str, default='WHITE', help='Ethnicity')
    parser.add_argument('--insurance', type=str, default='Medicare', help='Insurance type')
    parser.add_argument('--charlson', type=int, default=0, help='Charlson score')
    parser.add_argument('--prior_admits', type=int, default=0, help='Prior admissions (6 months)')
    parser.add_argument('--los', type=float, default=3.0, help='Length of stay (days)')
    
    # Diagnosis flags
    parser.add_argument('--heart_failure', action='store_true', help='Heart failure diagnosis')
    parser.add_argument('--copd', action='store_true', help='COPD diagnosis')
    parser.add_argument('--diabetes', action='store_true', help='Diabetes diagnosis')
    parser.add_argument('--renal', action='store_true', help='Renal failure diagnosis')
    
    # Hospital course
    parser.add_argument('--icu', action='store_true', help='ICU stay')
    parser.add_argument('--weekend', action='store_true', help='Weekend discharge')
    parser.add_argument('--medications', type=int, default=5, help='Number of medications')
    parser.add_argument('--diagnoses', type=int, default=3, help='Number of diagnoses')
    
    # Labs
    parser.add_argument('--creatinine', type=float, help='Creatinine (mg/dL)')
    parser.add_argument('--hemoglobin', type=float, help='Hemoglobin (g/dL)')
    
    # Clinical note
    parser.add_argument('--note', type=str, help='Path to clinical note file or note text')
    
    # Output
    parser.add_argument('--output', type=str, help='Save results to JSON file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Load clinical note if provided
    clinical_note = None
    if args.note:
        if Path(args.note).exists():
            with open(args.note, 'r') as f:
                clinical_note = f.read()
        else:
            clinical_note = args.note
    
    # Prepare patient data
    patient_data = {
        'age': args.age,
        'gender': args.gender,
        'ethnicity': args.ethnicity,
        'insurance': args.insurance,
        'charlson_score': args.charlson,
        'prior_admissions_180d': args.prior_admits,
        'los_days': args.los,
        'dx_heart_failure': args.heart_failure,
        'dx_copd': args.copd,
        'dx_diabetes': args.diabetes,
        'dx_renal_failure': args.renal,
        'had_icu_stay': args.icu,
        'discharge_weekend': args.weekend,
        'n_medications': args.medications,
        'n_diagnoses': args.diagnoses,
        'lab_creatinine': args.creatinine,
        'lab_hemoglobin': args.hemoglobin,
        'clinical_note': clinical_note
    }
    
    # Make prediction
    print("\n" + "="*60)
    print("TRANCE Readmission Prediction")
    print("="*60)
    
    if args.verbose:
        print("\nPatient Data:")
        for key, value in patient_data.items():
            if value is not None and key != 'clinical_note':
                print(f"  {key}: {value}")
        if clinical_note:
            print(f"  clinical_note: {len(clinical_note)} characters")
    
    print("\nGenerating prediction...")
    
    predictor = ReadmissionPredictor()
    result = predictor.predict(**patient_data)
    
    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"\n  Risk Score:       {result['risk_score']:.1f}%")
    print(f"  Risk Level:       {result['risk_level']}")
    print(f"  Risk Category:    {result['risk_category']}")
    print(f"  Probability:      {result['risk_probability']:.4f}")
    print(f"  Model Version:    {result['model_version']}")
    
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    print(f"\nOut of 100 similar patients:")
    print(f"  - {int(result['risk_score'])} would be readmitted within 30 days")
    print(f"  - {int(100 - result['risk_score'])} would NOT be readmitted")
    
    print("\n" + "="*60)
    print("RECOMMENDED ACTIONS")
    print("="*60)
    
    if result['risk_category'] == 'critical':
        print("""
  URGENT - High Priority Intervention:
  ✓ Schedule follow-up within 24-48 hours
  ✓ Coordinate with case management immediately
  ✓ Ensure clear medication reconciliation
  ✓ Assess home support and care needs
  ✓ Consider transitional care program enrollment
        """)
    elif result['risk_category'] == 'high':
        print("""
  MODERATE - Enhanced Follow-up:
  ✓ Schedule follow-up within 48-72 hours
  ✓ Provide detailed discharge instructions
  ✓ Verify medication understanding
  ✓ Check access to follow-up care
        """)
    elif result['risk_category'] == 'moderate':
        print("""
  STANDARD - Routine Follow-up:
  ✓ Schedule follow-up within 7-14 days
  ✓ Provide standard discharge education
  ✓ Monitor for any complications
        """)
    else:
        print("""
  LOW RISK - Standard Care:
  ✓ Routine discharge protocol
  ✓ Standard follow-up timing
  ✓ Patient education materials
        """)
    
    print("\n" + "="*60)
    
    # Save to file if requested
    if args.output:
        import json
        output_data = {
            'patient_data': {k: v for k, v in patient_data.items() if k != 'clinical_note'},
            'prediction': result,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✓ Results saved to: {args.output}")


if __name__ == "__main__":
    main()