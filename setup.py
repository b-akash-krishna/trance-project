"""
TRANCE Project Setup Script
Creates directory structure and initial configuration
"""

import os
import json
from pathlib import Path

def create_project_structure():
    """Create the complete TRANCE project directory structure"""
    
    # Define project structure
    directories = [
        'data/raw',
        'data/processed',
        'data/embeddings',
        'notebooks',
        'src/data_processing',
        'src/models',
        'src/evaluation',
        'src/dashboard',
        'configs',
        'outputs/models',
        'outputs/figures',
        'outputs/results',
        'docs',
        'tests'
    ]
    
    # Create directories
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}")
    
    # Create __init__.py files for Python packages
    src_dirs = ['src', 'src/data_processing', 'src/models', 'src/evaluation', 'src/dashboard']
    for src_dir in src_dirs:
        init_file = Path(src_dir) / '__init__.py'
        init_file.touch()
        print(f"✓ Created: {init_file}")
    
    return directories

def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """# Data files
data/raw/*
data/processed/*
data/embeddings/*
*.csv
*.db
*.sqlite
*.parquet
*.h5
*.hdf5

# Model files
outputs/models/*
*.pkl
*.joblib
*.pt
*.pth

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv

# Jupyter Notebooks
.ipynb_checkpoints
*/.ipynb_checkpoints/*

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
mlruns/

# Secrets
.env
secrets.json
credentials.json

# Large files
*.zip
*.tar.gz
*.7z

# But keep directory structure
!data/raw/.gitkeep
!data/processed/.gitkeep
!data/embeddings/.gitkeep
!outputs/models/.gitkeep
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    print("✓ Created: .gitignore")

def create_requirements():
    """Create requirements.txt with all necessary packages"""
    requirements = """# Data Processing
pandas==2.0.3
numpy==1.24.3
pyarrow==12.0.1
sqlalchemy==2.0.19
psycopg2-binary==2.9.6

# Machine Learning
scikit-learn==1.3.0
lightgbm==4.0.0
xgboost==1.7.6
optuna==3.2.0

# NLP & Transformers
transformers==4.31.0
torch==2.0.1
sentencepiece==0.1.99
tokenizers==0.13.3

# Evaluation & Interpretability
shap==0.42.1
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0

# Experiment Tracking
mlflow==2.5.0

# Dashboard
streamlit==1.25.0

# Utilities
tqdm==4.65.0
python-dotenv==1.0.0
pyyaml==6.0.1
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    print("✓ Created: requirements.txt")

def create_config():
    """Create initial configuration file"""
    config = {
        "project": {
            "name": "TRANCE",
            "version": "1.0.0",
            "description": "Temporal Readmission Analysis with Neural Clinical Embeddings"
        },
        "data": {
            "mimic_demo_url": "https://physionet.org/content/mimiciii-demo/1.4/",
            "cohort": {
                "min_age": 18,
                "min_los_hours": 24,
                "readmission_window_days": 30
            }
        },
        "model": {
            "embedding_model": "luqh/ClinicalT5-large",
            "embedding_dim": 768,
            "max_text_length": 512,
            "chunk_overlap": 128
        },
        "training": {
            "test_size": 0.2,
            "calibration_size": 0.1,
            "random_state": 42,
            "lgbm_params": {
                "objective": "binary",
                "metric": "auc",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.9
            }
        },
        "paths": {
            "raw_data": "data/raw",
            "processed_data": "data/processed",
            "embeddings": "data/embeddings",
            "models": "outputs/models",
            "figures": "outputs/figures"
        }
    }
    
    config_path = Path('configs/config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Created: {config_path}")
    
    return config

def create_readme():
    """Create comprehensive README"""
    readme_content = """# TRANCE: Temporal Readmission Analysis with Neural Clinical Embeddings

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
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

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
"""
    
    with open('README.md', 'w', encoding="utf-8") as f:
        f.write(readme_content)
    print("✓ Created: README.md")

def create_gitkeep_files(directories):
    """Create .gitkeep files to preserve empty directories"""
    empty_dirs = ['data/raw', 'data/processed', 'data/embeddings', 'outputs/models']
    for directory in empty_dirs:
        gitkeep = Path(directory) / '.gitkeep'
        gitkeep.touch()
        print(f"✓ Created: {gitkeep}")

def main():
    """Main setup function"""
    print("=" * 50)
    print("🏥 TRANCE Project Setup")
    print("=" * 50)
    print()
    
    # Create structure
    print("📁 Creating project structure...")
    directories = create_project_structure()
    print()
    
    # Create configuration files
    print("⚙️  Creating configuration files...")
    create_gitignore()
    create_requirements()
    create_config()
    create_readme()
    create_gitkeep_files(directories)
    print()
    
    # Summary
    print("=" * 50)
    print("✅ Setup Complete!")
    print("=" * 50)
    print()
    print("Next steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Download MIMIC-III demo data")
    print("3. Run notebooks/01_data_download.ipynb")
    print()
    print("Happy coding! 🚀")

if __name__ == "__main__":
    main()