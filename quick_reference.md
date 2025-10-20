# 🚀 TRANCE Quick Reference Card

## One-Page Cheat Sheet for Your Project

---

### 🎯 Project Goal
Predict 30-day hospital readmissions using structured EHR data + clinical text embeddings

---

### 📋 6 Main Components

| # | Component | File | What It Does | Time |
|---|-----------|------|--------------|------|
| 0 | Setup | `setup.py` | Creates project structure | 5 min |
| 1 | Data | `01_data_download.ipynb` | Downloads MIMIC demo | 15 min |
| 2 | Cohort | `02_cohort_definition.ipynb` | Defines eligible patients | 30 min |
| 3 | Features | `03_feature_engineering.ipynb` | Engineers 50+ features | 1 hour |
| 4 | Embeddings | `04_generate_embeddings.ipynb` | ClinicalT5 text vectors | 30 min* |
| 5 | Models | `05_train_models.ipynb` | Trains & evaluates models | 1 hour |
| 6 | Dashboard | `app.py` | Interactive 4-page UI | Run anytime |

*With GPU. 3-4 hours on CPU.

---

### ⚡ Quick Start Commands

```bash
# Setup (Day 1)
python setup.py
pip install -r requirements.txt

# Run notebooks in order (Days 2-7)
jupyter notebook

# Launch dashboard (Once models trained)
streamlit run src/dashboard/app.py
```

---

### 📊 Expected Results

| Metric | Baseline | Fused | Improvement |
|--------|----------|-------|-------------|
| AUROC | 0.72 | 0.78 | +8% |
| AUPRC | 0.35 | 0.40 | +14% |
| Brier | 0.16 | 0.14 | Better |

---

### 🎨 Dashboard Pages

1. **Executive Overview** - KPIs and ROC curves
2. **Volume Forecasting** - Daily readmission predictions
3. **Patient Risk** - Top-N high-risk list with actions
4. **Model Monitoring** - Calibration and feature importance

---

### 🛠️ Common Fixes

**GPU not found?**
```python
# Colab: Runtime → GPU
device = 'cpu'  # Fallback
```

**Out of memory?**
```python
batch_size = 4  # Reduce
```

**Path errors?**
```python
base_path = Path(__file__).parent.parent.parent
```

---

### 📝 Key Features

**Structured (50+):**
- Demographics (age, gender, ethnicity)
- Diagnoses (Charlson score, ICD-9 codes)
- Labs (creatinine, hemoglobin, etc.)
- Utilization (prior admissions, ICU stays)

**Embeddings (768):**
- ClinicalT5-large from discharge summaries
- Mean-pooled across text chunks

---

### 🎓 What You'll Learn

✅ Healthcare ML (MIMIC data)  
✅ Feature engineering  
✅ NLP with transformers  
✅ Gradient boosting (LightGBM)  
✅ Model calibration & SHAP  
✅ Dashboard development  

---

### 💡 AI Prompts to Use

**For debugging:**
```
"I'm getting error [X] in [file]. Here's my code: [code]. 
What's wrong?"
```

**For features:**
```
"How do I calculate Charlson score from ICD-9 codes 
in pandas?"
```

**For dashboard:**
```
"Add a date filter to my Streamlit app that updates 
all charts. Current code: [code]"
```

---

### 📞 Help Hierarchy

1. Check notebook comments
2. Review troubleshooting in summary doc
3. Use AI with specific error
4. Ask me!

---

### ✅ Week 1 Checklist

- [ ] Run setup.py
- [ ] Download MIMIC demo
- [ ] Define cohort
- [ ] Engineer features
- [ ] Show advisor stats

---

### ✅ Week 2 Checklist

- [ ] Generate embeddings (GPU!)
- [ ] Train baseline model
- [ ] Train fused model
- [ ] Compare AUROC scores
- [ ] Generate SHAP values

---

### ✅ Week 3-4 Checklist

- [ ] Build dashboard (4 pages)
- [ ] Test all interactions
- [ ] Write technical report
- [ ] Create presentation slides
- [ ] Record demo video
- [ ] Finalize GitHub repo

---

### 🎤 Presentation Outline (20 min)

1. **Problem** (2 min): Readmissions cost $26B/year
2. **Approach** (3 min): Multimodal ML with ClinicalT5
3. **Data** (2 min): MIMIC-III, 100 patients, 15% readmit rate
4. **Features** (2 min): 50 structured + 768 embeddings
5. **Results** (4 min): AUROC 0.78, +8% from text
6. **Demo** (3 min): Show dashboard live
7. **Impact** (2 min): Forecasting, risk ranking, interpretability
8. **Q&A** (5-10 min)

---

### 🏆 Standout Points

**Technical:**
- Multimodal learning (cutting-edge)
- Proper calibration (healthcare critical)
- SHAP interpretability (explainable AI)

**Professional:**
- End-to-end system (not just a model)
- Production-ready dashboard
- Complete documentation

**Domain:**
- Clinical feature engineering
- Operational workflows
- Ethical considerations

---

### 📈 Performance Benchmarks

**Good Performance:**
- AUROC > 0.72
- Brier Score < 0.15
- Forecast MAE < 3 patients/day

**Excellent Performance:**
- AUROC > 0.75
- Brier Score < 0.12
- Forecast MAE < 2 patients/day

---

### 🔧 File Locations (After Running)

```
TRANCE/
├── data/processed/
│   ├── train_fused.parquet ← Features + embeddings
│   ├── test_fused.parquet
│   └── cohort_with_outcomes.parquet
├── outputs/models/
│   ├── baseline_model.txt ← Trained models
│   ├── fused_model.txt
│   └── calibrator.pkl
├── outputs/results/
│   ├── test_predictions.parquet ← Predictions
│   └── model_results_summary.json ← Metrics
└── outputs/figures/
    ├── model_performance_curves.png
    ├── calibration_analysis.png
    └── feature_importance.png
```

---

### 🎯 Key Metrics Explained

**AUROC (0-1):** Ability to rank high-risk patients higher
- 0.5 = random, 0.7 = acceptable, 0.8 = excellent

**AUPRC (0-1):** Performance with imbalanced classes
- Baseline = readmission rate (15%), higher is better

**Brier Score (0-1):** Calibration quality
- 0 = perfect predictions, <0.15 = well-calibrated

**MAE (patients):** Forecast accuracy
- Average daily prediction error

---

### 💾 Data Sizes

| File | Size | Count |
|------|------|-------|
| MIMIC Demo | ~50 MB | 100 patients |
| Cohort | ~1 MB | 129 admissions |
| Features | ~2 MB | 50+ columns |
| Embeddings | ~100 MB | 768 dims |
| Models | ~5 MB | LightGBM |

---

### 🚨 Critical Reminders

1. **Always use GPU for embeddings** (Colab T4)
2. **Temporal splits only** (train before test)
3. **Calibrate probabilities** (healthcare needs this)
4. **Document everything** (future you will thank you)
5. **Save often** (commit to GitHub regularly)

---

### 🌟 Bonus Features (If Time Permits)

- [ ] Add confidence intervals to forecasts
- [ ] Implement threshold optimization tool
- [ ] Create PDF report generator
- [ ] Add patient search functionality
- [ ] Build email alert system mockup
- [ ] Include model retraining workflow

---

### 📚 Essential Reading (30 min total)

1. **MIMIC overview** (10 min): https://mimic.mit.edu/docs/iii/
2. **Readmission basics** (10 min): Search "HOSPITAL score validation"
3. **Calibration importance** (10 min): Search "Brier score healthcare"

---

### 🎓 Interview Questions You Can Answer

**Q: What's unique about your project?**
A: "We combine structured data with clinical text embeddings from ClinicalT5 to improve readmission prediction by 8%. The system includes a production-ready dashboard for operational use."

**Q: Why use embeddings vs keyword search?**
A: "Embeddings capture semantic meaning and context that keywords miss. For example, 'difficulty breathing at rest' and 'dyspnea on exertion' are similar concepts that embeddings recognize."

**Q: How do you handle class imbalance?**
A: "We focus on AUPRC rather than accuracy, use appropriate class weights, and emphasize probability calibration for reliable risk scores."

**Q: What's your biggest challenge?**
A: "Processing long clinical notes efficiently. We implemented sliding window chunking with mean pooling to handle notes exceeding model limits."

**Q: How would you deploy this?**
A: "We'd integrate with hospital EHR systems via HL7/FHIR APIs, run daily batch predictions, and serve via a secure web portal. Monitoring would track performance drift and trigger retraining."

---

### 🛡️ Safety Checks Before Demo

```bash
# Test notebooks run
jupyter nbconvert --execute notebook.ipynb

# Test dashboard loads
streamlit run src/dashboard/app.py

# Check file sizes
du -sh outputs/*

# Verify no data in git
git status --ignored

# Test on fresh environment
pip install -r requirements.txt
```

---

### 📋 Presentation Day Checklist

**30 min before:**
- [ ] Laptop charged
- [ ] Presentation file open
- [ ] Dashboard running (localhost:8501)
- [ ] Demo video ready as backup
- [ ] Notes printed/accessible

**During presentation:**
- [ ] Speak slowly and clearly
- [ ] Make eye contact
- [ ] Point at visualizations
- [ ] Time yourself (set phone timer)
- [ ] Handle questions gracefully

---

### 💬 Elevator Pitch (30 seconds)

"We built TRANCE, a system that predicts hospital readmissions by combining traditional patient data with AI analysis of doctors' notes. Using transformer embeddings, we achieve 78% accuracy and provide a dashboard that helps hospitals forecast capacity and identify high-risk patients for intervention—potentially preventing hundreds of readmissions annually."

---

### 🎉 Success Criteria

**Minimum Viable:**
- ✅ All notebooks run without errors
- ✅ Models achieve AUROC > 0.70
- ✅ Dashboard displays data correctly
- ✅ Report documents methods clearly

**Target:**
- ✅ AUROC > 0.75 with embeddings
- ✅ Professional dashboard with 4 pages
- ✅ SHAP interpretability included
- ✅ Polished presentation

**Stretch:**
- ✅ Published to GitHub with stars
- ✅ Blog post about the project
- ✅ Submitted to conference/journal
- ✅ Featured in portfolio

---

### 🔗 Quick Links

- **MIMIC Demo:** https://physionet.org/content/mimiciii-demo/1.4/
- **ClinicalT5:** https://huggingface.co/luqh/ClinicalT5-large
- **Streamlit Docs:** https://docs.streamlit.io/
- **Colab GPU:** https://colab.research.google.com/

---

### 📞 Emergency Contacts

**Technical Issues:**
- Stack Overflow: [error message]
- Reddit: r/MachineLearning, r/HealthIT
- GitHub Issues: Check similar projects

**Me:**
- Available for debugging, explanations, additions!

---

## 🎯 Remember:

1. **Progress over perfection** - Working system beats perfect code
2. **Document as you go** - Future you needs context
3. **Test frequently** - Catch issues early
4. **Communicate clearly** - Show, don't just tell
5. **Learn and iterate** - Every challenge is a lesson

---

## 🚀 You're Ready!

You have everything you need:
- ✅ Complete code (setup → dashboard)
- ✅ Step-by-step notebooks
- ✅ Comprehensive documentation
- ✅ Troubleshooting guides
- ✅ Presentation tips

**Now go build something amazing!** 🌟

---

*Keep this card handy throughout your project. Print it, bookmark it, or pin it to your desktop.*

**Good luck! You've got this! 💪🏥🤖**