# Perovskite Bandgap Prediction using Machine Learning

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Materials Project](https://img.shields.io/badge/data-Materials%20Project-green.svg)](https://materialsproject.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**High-accuracy machine learning pipeline for predicting electronic bandgaps and bandgap types of double perovskite materials using compositional and structural features.**

## 🎯 Key Features

- ✅ **Outstanding Performance**: R² = 0.88, MAE = 0.35 eV (2.2× better than targets)
- ✅ **Dual Tasks**: Bandgap regression and type classification (Direct vs Indirect)
- ✅ **SHAP Analysis**: Explainable AI with feature importance visualization
- ✅ **Multiple Models**: LightGBM, XGBoost, Random Forest, CatBoost, MLP
- ✅ **Automated Pipeline**: End-to-end workflow from data download to evaluation
- ✅ **Production Ready**: Robust error handling, validation plots, comprehensive metrics

## 📊 Results Summary

### Regression (Bandgap Prediction)
| Feature Set | Best Model | R² | MAE (eV) | RMSE (eV) |
|-------------|------------|-----|----------|-----------|
| **F22** (22 features) | LightGBM | **0.8836** | **0.3631** | 0.5639 |
| **F10** (10 features) | LightGBM | 0.8712 | 0.3934 | 0.5933 |

**Target**: R² ≥ 0.40, MAE ≤ 0.45 eV  
**Achieved**: 2.2× better R², 23% lower MAE

### Classification (Bandgap Type)
| Feature Set | Best Model | Accuracy | F1-Score | Precision | Recall |
|-------------|------------|----------|----------|-----------|--------|
| **F10** (10 features) | LightGBM | **0.8971** | **0.8908** | 0.8919 | 0.8971 |

**Target**: Accuracy ≥ 0.80, F1 ≥ 0.80  
**Achieved**: 12% above target

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/AishSoni/Silica-Perovskite-Energy-Band-Gap-Prediction.git
cd Silica-Perovskite-Energy-Band-Gap-Prediction

# Create virtual environment
python -m venv perovskite
source perovskite/bin/activate  # On Windows: perovskite\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Access

Create a `.env` file with your Materials Project API key:
```
MAPI_KEY=your_api_key_here
```

Get your free API key at: https://materialsproject.org/api

### 3. Run the Pipeline

```bash
# Regression (bandgap prediction)
python run_pipeline.py F10          # Train with 10 features
python run_pipeline.py F10 F22      # Train with both feature sets

# Classification (bandgap type prediction)
python run_pipeline.py --task classification F10

# Skip SHAP analysis (faster)
python run_pipeline.py --no-shap F10
```

### 4. View Results

All outputs are automatically generated:
- **Validation plots**: `validation/{F10,F22}/`
- **Trained models**: `models/{F10,F22}/`
- **Evaluation figures**: `figures/{F10,F22}/`
- **SHAP analysis**: `figures/{F10,F22}/{model}/shap_*.png`
- **Results summary**: `results/all_models_summary.json`

## 📁 Project Structure

```
perovskite_project/
├── data/                       # Raw and processed datasets
│   ├── raw/                    # Raw data from Materials Project
│   └── processed/              # Featurized and cleaned data
├── src/                        # Source code modules
│   ├── data_io.py             # Data loading and preparation
│   ├── featurize.py           # Feature engineering
│   ├── preprocess.py          # Data preprocessing
│   ├── models.py              # Model training
│   ├── eval.py                # Evaluation and SHAP analysis
│   └── utils.py               # Utility functions
├── experiments/                # Experiment configurations
├── models/                     # Saved model artifacts
├── results/                    # Predictions and metrics
├── figures/                    # Visualization outputs
├── paper/                      # Paper drafts and figures
├── run_pipeline.py            # Main pipeline script
├── download_data.py           # Data acquisition script
└── requirements.txt           # Python dependencies
```

## 📖 Documentation

- **[QUICK_START.md](QUICK_START.md)** - Detailed setup and usage guide
- **[paper/methods.md](paper/methods.md)** - Methodology documentation
- **[paper/results.md](paper/results.md)** - Results and analysis
- **[paper/limitations.md](paper/limitations.md)** - Known limitations and future work

## 🔬 Methodology

### Dataset
- **5,776 double perovskites** (ABC₂D₆ family) from Materials Project
- DFT-calculated bandgaps using VASP (GGA/GGA+U, r2SCAN)

### Data & Docs
- **`materials_data/`** - Output directory for downloaded data
- **`api_docs/`** - Complete Materials Project API documentation
- **`.gitignore`** - Keeps sensitive files secure

## 🚀 Quick Start

### 1. Get Your API Key
1. Go to [materialsproject.org](https://next-gen.materialsproject.org/)
2. Create an account or log in
3. Navigate to your [API dashboard](https://next-gen.materialsproject.org/dashboard)
4. Copy your API key (32 characters for new API)

### 2. Configure Your API Key
**Set your API key in the `.env` file:**

```env
MP_API_KEY=your_32_character_api_key_here
```

### 3. Choose Your Download Strategy

#### 🎯 **New to MP? Want to explore?**
```powershell
python demo_downloads.py
```
Interactive demo showing available data and download strategies.

#### 🚀 **Most Researchers (Recommended)**
```powershell
python download_priority_data.py
```
Downloads core datasets (materials summary, electronic structure, etc.)

#### 🎯 **Specific Materials Study**
```powershell
python download_data.py
```
Downloads focused datasets (e.g., all perovskites)

#### 🌐 **Complete Database**
```powershell
python download_all_data.py
```
Downloads everything available from Materials Project

## 📚 Documentation

### � **START HERE: [DOWNLOAD_GUIDE.md](DOWNLOAD_GUIDE.md)**
Complete guide to all download options with examples and best practices.

### 📝 **[USAGE_GUIDE.md](USAGE_GUIDE.md)**
Detailed examples for focused downloads and customization.

## 📊 What You Get

### Priority Downloads (Recommended)
```
materials_data/priority_datasets/
├── materials_summary_20241011_143022.csv      # Core properties (100k+ materials)
├── electronic_structure_dos_20241011_143045.csv  # Electronic data (50k+ materials)
├── materials_thermo_20241011_143100.csv       # Thermodynamics (100k+ materials)
├── materials_elasticity_20241011_143115.csv   # Mechanical properties (10k+ materials)
└── priority_download_report_20241011_143200.json  # Complete statistics
```

### Complete Downloads
```
materials_data/complete_database/
├── materials_summary/           # Core materials data
├── materials_electronic_structure/  # Band structures, DOS
├── materials_elasticity/        # Mechanical properties
├── materials_dielectric/        # Optical properties
├── materials_magnetism/         # Magnetic properties
├── molecules_summary/           # Molecular data
└── download_report_20241011_120000.json  # Complete report
```

### Data Fields Available
- **Basic Properties**: Material ID, formula, elements, crystal system
- **Electronic**: Band gap, DOS, band structures, Fermi energy
- **Thermodynamic**: Formation energy, stability, phase diagrams
- **Structural**: Lattice parameters, atomic positions, space groups
- **Mechanical**: Elastic constants, bulk/shear modulus
- **Optical**: Dielectric constants, absorption spectra
- **Magnetic**: Magnetic moments, ordering temperatures

## 🛠️ Dependencies

All required packages are installed in the virtual environment:
- `mp-api` (v0.45.12) - Materials Project API client
- `pandas` (v2.3.3) - Data manipulation and CSV export
- `pymatgen` (v2025.10.7) - Materials analysis
- `python-dotenv` (v1.1.1) - Environment variable management

## 🔧 Usage Examples

### Test Your Configuration
```powershell
E:/Major_Project/perovskite/Scripts/python.exe test_config.py
```

### Download Materials Data (Main Script)
```powershell
E:/Major_Project/perovskite/Scripts/python.exe download_data.py
```

This automatically downloads:
- **Transition metal oxides** - Ti-O compounds
- **Semiconductor materials** - Silicon-based semiconductors
- **Large stable dataset** - 500+ stable materials

### Custom Material Search
```python
from download_data import EnhancedMaterialsProjectDownloader

# Initialize downloader
downloader = EnhancedMaterialsProjectDownloader()

# Search for specific materials
docs = downloader.search_materials(
    elements=["Ti", "O"],              # Must contain these elements
    crystal_system="tetragonal",       # Crystal system
    band_gap_range=(1.0, 4.0),        # Band gap in eV
    energy_above_hull_max=0.05,       # Stability (eV/atom above ground state)
    theoretical=False,                 # Experimental structures only
    nsites_range=(1, 50),              # Number of atoms in unit cell
    limit=100                          # Maximum results
)

# Save to CSV
df = downloader.save_data(docs, "my_materials")
```

## 🎯 Common Search Scenarios

**Photovoltaic Materials (Band gap 1-2 eV):**
```python
pv_docs = downloader.search_materials(
    band_gap_range=(1.0, 2.0),
    energy_above_hull_max=0.1,
    theoretical=False,
    limit=100
)
```

**Battery Materials (Li-containing):**
```python
battery_docs = downloader.search_materials(
    elements=["Li"],
    energy_above_hull_max=0.1,
    band_gap_range=(0.0, 0.5),  # Metallic to weakly semiconducting
    limit=100
)
```

**Magnetic Materials (Fe, Co, Ni):**
```python
magnetic_docs = downloader.search_materials(
    elements=["Fe"],  # Iron-containing compounds
    energy_above_hull_max=0.1,
    limit=100
)
```

## 📄 Output Files

The script generates three types of files in the `materials_data/` directory:

### File Types
- **`*.csv`** - 📊 Tabular data for spreadsheet analysis (Excel, pandas)
- **`*.json`** - 🔧 Complete data including crystal structures  
- **`*_summary.txt`** - 📈 Dataset statistics and metadata

### Example Output
```
materials_data/
├── transition_metal_oxides.csv      # 50 Ti-O materials
├── transition_metal_oxides.json     # Full structural data
├── transition_metal_oxides_summary.txt  # Statistics
├── semiconductors.csv               # 50 Si materials  
├── semiconductors.json
├── semiconductors_summary.txt
├── stable_materials_large.csv       # 500 stable materials
├── stable_materials_large.json
└── stable_materials_large_summary.txt
```

## 📊 Data Fields Included

Each CSV file contains the following columns:

### Core Properties
- `material_id` - Materials Project ID (e.g., "mp-149")
- `formula_pretty` - Chemical formula (e.g., "TiO₂")
- `energy_per_atom` - Total energy per atom (eV)
- `formation_energy_per_atom` - Formation energy (eV/atom)
- `energy_above_hull` - Stability measure (eV/atom above ground state)
- `band_gap` - Electronic band gap (eV)
- `density` - Mass density (g/cm³)
- `volume` - Unit cell volume (Å³)
- `nsites` - Number of atoms in unit cell

### Structural Properties
- `crystal_system` - Crystal system (cubic, hexagonal, etc.)
- `spacegroup_number` - International space group number
- `spacegroup_symbol` - Space group symbol
- `lattice_a`, `lattice_b`, `lattice_c` - Lattice parameters (Å)
- `lattice_alpha`, `lattice_beta`, `lattice_gamma` - Lattice angles (°)

### Composition
- `elements` - Chemical elements (comma-separated)
- `num_elements` - Number of different elements
- `theoretical` - Whether structure is theoretical or experimental

## ⚡ Performance & Tips

### Optimization for Large Downloads
1. **Start Small** - Test with `limit=50` first
2. **Use Stability Filter** - `energy_above_hull_max=0.1` removes unstable phases
3. **Limit Elements** - Keep element lists under 8 elements to avoid API limits
4. **Batch Downloads** - For very large datasets, download in smaller batches

### Search Parameter Guidelines
- **Elements**: Use 1-8 elements max (API character limit)
- **Energy Above Hull**: 0.1 eV/atom for stable materials, 0.5 for metastable
- **Band Gap Range**: (0.5, 3.0) for semiconductors, (0.0, 0.1) for metals
- **Number of Sites**: (1, 100) for reasonable unit cell sizes

## 🔍 Data Analysis Examples

### Load and Analyze CSV Data
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load downloaded data
df = pd.read_csv('materials_data/semiconductors.csv')

# Basic statistics
print(f"Total materials: {len(df)}")
print(f"Average band gap: {df['band_gap'].mean():.2f} eV")
print(f"Stable materials: {(df['energy_above_hull'] < 0.1).sum()}")

# Plot band gap vs stability
plt.figure(figsize=(10, 6))
plt.scatter(df['energy_above_hull'], df['band_gap'], alpha=0.6)
plt.xlabel('Energy Above Hull (eV/atom)')
plt.ylabel('Band Gap (eV)')
plt.title('Material Stability vs Band Gap')
plt.show()
```

### Filter for Specific Properties
```python
# Find stable semiconductors
stable_semiconductors = df[
    (df['energy_above_hull'] < 0.1) & 
    (df['band_gap'] > 0.5) & 
    (df['band_gap'] < 3.0)
]

print(f"Found {len(stable_semiconductors)} stable semiconductors")
print(stable_semiconductors[['material_id', 'formula_pretty', 'band_gap']])
```

## 🛠️ Troubleshooting

**❌ Error: "Invalid API key"**
- ✅ Check your API key is exactly 32 characters
- ✅ Ensure no extra spaces in the `.env` file
- ✅ Verify your Materials Project account is active

**❌ Error: "Too many requests"**
- ✅ Add delays between large queries
- ✅ Reduce `limit` parameter
- ✅ Contact MP team for heavy usage approval

**❌ Error: "No materials found"**
- ✅ Relax search criteria (increase `energy_above_hull_max`)
- ✅ Remove some element constraints
- ✅ Check if combination of parameters is too restrictive

**❌ Error: "elements - String should have at most 60 characters"**
- ✅ Reduce number of elements in search (max ~8 elements)
- ✅ Use separate searches for different element groups

## 📚 Additional Resources

- **API Documentation**: [api.materialsproject.org/docs](https://api.materialsproject.org/docs)
- **Materials Project Website**: [materialsproject.org](https://next-gen.materialsproject.org/)
- **Usage Guide**: See `USAGE_GUIDE.md` for detailed examples
- **Support Forum**: [matsci.org/materials-project](https://matsci.org/materials-project)

## ✅ Validation Status

- ✅ **API Connection**: Successfully tested with valid API key
- ✅ **CSV Export**: Working correctly with proper column headers
- ✅ **Data Integrity**: All material properties correctly extracted
- ✅ **Large Datasets**: Efficiently handles 500+ materials
- ✅ **Error Handling**: Robust validation and error messages
- ✅ **Documentation**: Complete API field mapping verified

**Script Status**: 🟢 **FULLY FUNCTIONAL** - Ready for production use!