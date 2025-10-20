# TRANCE: Temporal Readmission Analysis with Neural Clinical Embeddings

## Project Overview
A multimodal machine learning system for predicting 30-day hospital readmissions using structured EHR data and clinical text embeddings.

## Team Members
- Member 1: Data Engineer
- Member 2: ML Engineer  
- Member 3: Full-Stack Developer

## Project Structure
```
TRANCE/
├── data/                  # Data storage (not in git)
├── notebooks/             # Jupyter notebooks for exploration
├── src/                   # Source code
│   ├── data_processing/   # Data extraction & feature engineering
│   ├── models/            # Model training & evaluation
│   ├── evaluation/        # Calibration & metrics
│   └── dashboard/         # Streamlit dashboard
├── configs/               # Configuration files
├── outputs/               # Models, figures, results
└── docs/                  # Documentation
```

## Setup Instructions

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. MIMIC-III Demo Data
```bash
# Download from PhysioNet (no credentials needed for demo)
# Instructions in notebooks/01_data_download.ipynb
```

### 3. Run Pipeline
```bash
# Step 1: Data extraction
python src/data_processing/extract_cohort.py

# Step 2: Feature engineering
python src/data_processing/engineer_features.py

# Step 3: Generate embeddings
python src/data_processing/generate_embeddings.py

# Step 4: Train model
python src/models/train_model.py

# Step 5: Launch dashboard
streamlit run src/dashboard/app.py
```

## Key Features
- ✅ Multimodal learning (structured + text embeddings)
- ✅ Probability calibration for reliable forecasts
- ✅ SHAP interpretability
- ✅ Interactive dashboard for clinical decision support

## Tech Stack
- **Data**: MIMIC-III Demo, PostgreSQL
- **ML**: LightGBM, ClinicalT5, scikit-learn
- **Visualization**: Streamlit, Plotly
- **Tracking**: MLflow

## Current Status
- [ ] Phase 1: Data acquisition ⏳
- [ ] Phase 2: Feature engineering
- [ ] Phase 3: Model development
- [ ] Phase 4: Calibration
- [ ] Phase 5: Dashboard
- [ ] Phase 6: Documentation

## Resources
- [MIMIC-III Demo](https://physionet.org/content/mimiciii-demo/1.4/)
- [ClinicalT5 Paper](https://arxiv.org/abs/2210.07867)
- [Project Plan](docs/implementation_plan.md)

## License
MIT License - Academic use only
