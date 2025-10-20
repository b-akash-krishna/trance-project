# 🚀 TRANCE Project - Getting Started Guide

## Step-by-Step Setup (30 minutes)

### Phase 1: Environment Setup

#### Step 1: Create Project Directory
```bash
# Create main folder
mkdir TRANCE
cd TRANCE

# Copy the setup script I provided into setup.py
# Then run it:
python setup.py
```

This creates your complete project structure:
```
TRANCE/
├── data/              # All data files (not in git)
├── notebooks/         # Jupyter notebooks
├── src/              # Python source code
├── configs/          # Configuration files
├── outputs/          # Results and models
└── docs/             # Documentation
```

#### Step 2: Install Dependencies
```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

**Estimated time: 5-10 minutes** (depending on internet speed)

---

### Phase 2: Data Acquisition

#### Step 3: Download MIMIC-III Demo Data

The demo dataset is **publicly available** - no credentials needed!

**Option A: Using the Notebook (Recommended)**
```bash
# Launch Jupyter
jupyter notebook

# Open: notebooks/01_data_download.ipynb
# Run all cells (Shift + Enter)
```

**Option B: Manual Download**
1. Visit: https://physionet.org/content/mimiciii-demo/1.4/
2. Download these files to `data/raw/`:
   - ADMISSIONS.csv.gz
   - PATIENTS.csv.gz
   - DIAGNOSES_ICD.csv.gz
   - NOTEEVENTS.csv.gz
   - LABEVENTS.csv.gz
   - PRESCRIPTIONS.csv.gz

3. Extract them:
```bash
cd data/raw
gunzip *.gz
```

**Estimated time: 10 minutes**

---

### Phase 3: First Milestone - Data Exploration

#### Step 4: Run Data Exploration
```bash
# In Jupyter, open and run:
notebooks/01_data_download.ipynb
```

**What you'll get:**
- ✅ All demo data downloaded
- ✅ Basic statistics computed
- ✅ Readmission rate calculated
- ✅ Sample clinical notes viewed
- ✅ Summary saved to `outputs/results/`

**Estimated time: 5 minutes**

---

## Team Workflow with AI Tools

Since your team uses AI tools heavily, here's the optimal workflow:

### Role Distribution (3 people)

**Person 1: Data Pipeline Lead**
- Focus: Notebooks 01-03 (data extraction, cohorts, features)
- Tools: Use AI to help write SQL queries and pandas transformations
- Deliverable: Clean feature matrices

**Person 2: Model & ML Lead**  
- Focus: Notebooks 04-05 (embeddings, model training)
- Tools: Use AI to help with model code and hyperparameter tuning
- Deliverable: Trained models with SHAP analysis

**Person 3: Dashboard Lead**
- Focus: Streamlit dashboard (all pages)
- Tools: Use AI to help with Plotly visualizations and UI components
- Deliverable: Working web application

### Using AI Tools Effectively

**For Each Task:**
1. **Understand first**: Read the notebook/script I provide
2. **Ask AI**: "Explain this code section: [paste code]"
3. **Modify**: "Adapt this code to [your specific need]"
4. **Debug**: "I'm getting this error: [error message]. Here's my code: [code]"
5. **Enhance**: "Add [feature] to this code: [code]"

**Example Prompts:**
```
"Explain how this SQL query works for extracting admissions"
"Add age filtering to this cohort selection code"
"Create a plotly chart showing readmission trends over time"
"Debug this pandas merge error: [error details]"
"Optimize this embedding generation loop for large datasets"
```

---

## Your First Week Goals

### Day 1-2: Setup & Exploration
- [ ] Run `setup.py` - create project structure
- [ ] Install all dependencies
- [ ] Download MIMIC demo data
- [ ] Run `01_data_download.ipynb` - understand the data
- [ ] Review exploration summary

### Day 3-4: Cohort Definition
- [ ] Run `02_cohort_definition.ipynb` (I'll provide next)
- [ ] Define inclusion/exclusion criteria
- [ ] Create readmission labels
- [ ] Validate cohort size and stats

### Day 5-7: Feature Engineering
- [ ] Run `03_feature_engineering.ipynb` (I'll provide)
- [ ] Extract demographic features
- [ ] Compute comorbidity scores
- [ ] Process lab values
- [ ] Create train/test splits

**End of Week 1 Deliverable:** 
Clean dataset ready for modeling (structured features + text notes)

---

## Quick Wins to Show Progress

### After Day 2:
**Show your advisor:**
- "We've set up the environment and explored the data"
- Share: `outputs/results/data_exploration_summary.json`
- Highlight: "Found 129 admissions, 15% readmission rate"

### After Day 4:
**Show your advisor:**
- "We've defined our cohort with clear criteria"
- Share: Cohort statistics table
- Highlight: "Excluding same-day readmissions and in-hospital deaths"

### After Day 7:
**Show your advisor:**
- "We have 50+ engineered features ready"
- Share: Feature importance preliminary plot
- Highlight: "Ready to start model training"

---

## Troubleshooting Common Issues

### Issue 1: Can't download MIMIC demo
**Solution:** The demo is public, no credentials needed. Use the direct URLs in notebook.

### Issue 2: Out of memory loading data
**Solution:** Use chunking:
```python
# Instead of:
df = pd.read_csv('large_file.csv')

# Do:
chunks = pd.read_csv('large_file.csv', chunksize=10000)
df = pd.concat([chunk for chunk in chunks])
```

### Issue 3: GPU not available for embeddings
**Solution:** 
- Free tier Colab has T4 GPU (sufficient!)
- Go to Runtime → Change runtime type → GPU
- Or use CPU (slower but works for demo data)

### Issue 4: Package conflicts
**Solution:**
```bash
# Create fresh environment
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

---

## Resources at Your Fingertips

### MIMIC-III Documentation
- Official Docs: https://mimic.mit.edu/docs/iii/
- Tables Guide: https://mimic.mit.edu/docs/iii/tables/
- SQL Examples: https://github.com/MIT-LCP/mimic-code/

### Python Libraries
- LightGBM Docs: https://lightgbm.readthedocs.io/
- SHAP Examples: https://shap.readthedocs.io/
- Streamlit Gallery: https://streamlit.io/gallery

### Ask Me Anything!
I'm here to help with:
- ❓ Understanding any code section
- 🐛 Debugging errors
- 🎨 Customizing visualizations
- 🚀 Optimizing performance
- 📊 Interpreting results

---

## Next Steps After This Guide

1. **Run the setup script** → Get project structure
2. **Download demo data** → Run notebook 01
3. **Ask me for notebook 02** → Cohort definition
4. **Keep building** → I'll provide each component as you progress

**Ready to start? Let's do this! 🎉**

---

## Quick Reference Commands

```bash
# Activate environment
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Launch Jupyter
jupyter notebook

# Run a Python script
python src/data_processing/script_name.py

# Launch dashboard (later)
streamlit run src/dashboard/app.py

# Check GPU in Colab
import torch
print(torch.cuda.is_available())
```

---

**Questions? Issues? Next steps?** Just ask! I'll provide the next notebook or help debug anything. 🚀