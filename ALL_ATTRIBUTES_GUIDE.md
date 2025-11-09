# All Attributes Download Feature

## Overview

The enhanced Materials Project downloader now supports downloading **ALL available attributes** for materials, not just the predefined subset of fields. This gives you access to the complete Materials Project database for comprehensive materials analysis.

## New Parameters

### `all_attributes` Parameter

- **`all_attributes=False`** (default): Downloads only the standard set of important fields
- **`all_attributes=True`**: Downloads EVERY available field from the Materials Project API

### Updated Methods

#### 1. `query_all_perovskite_materials(all_attributes=False)`

```python
# Standard fields only (faster, smaller files)
docs = downloader.query_all_perovskite_materials(all_attributes=False)

# ALL available attributes (comprehensive, larger files)
docs = downloader.query_all_perovskite_materials(all_attributes=True)
```

#### 2. `save_data(docs, filename, all_attributes=False)`

```python
# Save with standard DataFrame conversion
df = downloader.save_data(docs, "my_materials", all_attributes=False)

# Save preserving all available attributes
df = downloader.save_data(docs, "my_materials", all_attributes=True)
```

#### 3. `download_all_perovskite_attributes(limit=None)` - NEW!

Convenience method for downloading all Perovskite materials with all attributes:

```python
# Download first 50 Perovskites with all attributes (for testing)
df = downloader.download_all_perovskite_attributes(limit=50)

# Download ALL Perovskites with all attributes
df = downloader.download_all_perovskite_attributes()
```

## Usage Examples

### Example 1: Quick All-Attributes Download

```python
from download_data import EnhancedMaterialsProjectDownloader

downloader = EnhancedMaterialsProjectDownloader()

# Download 10 Perovskites with all attributes
df = downloader.download_all_perovskite_attributes(limit=10)
print(f"Downloaded {len(df.columns)} attributes for {len(df)} materials")
```

### Example 2: Compare Standard vs All Attributes

```python
# Standard fields
docs_std = downloader.query_all_perovskite_materials(all_attributes=False)
df_std = downloader.save_data(docs_std, "standard", all_attributes=False)

# All attributes
docs_all = downloader.query_all_perovskite_materials(all_attributes=True)
df_all = downloader.save_data(docs_all, "complete", all_attributes=True)

print(f"Standard: {len(df_std.columns)} columns")
print(f"Complete: {len(df_all.columns)} columns")
```

### Example 3: Search Any Materials with All Attributes

```python
# Search for titanium oxides with all available data
docs = downloader.search_materials(
    elements=['Ti', 'O'],
    # Omitting 'fields' parameter gets all fields
    limit=20
)

# Save with all attributes preserved
df = downloader.save_data(docs, "titanium_oxides_complete", all_attributes=True)
```

## What's Included in "All Attributes"

When `all_attributes=True`, you get access to:

### Core Properties
- `material_id`, `formula_pretty`, `formula_reduced`
- `structure`, `lattice parameters`, `volume`, `density`
- `elements`, `composition`, `nsites`

### Electronic Properties
- `band_gap`, `is_gap_direct`, `cbm`, `vbm`
- `dos`, `band_structure` (if available)
- `efermi`, `electronic_structure`

### Thermodynamic Properties
- `formation_energy_per_atom`, `energy_per_atom`
- `energy_above_hull`, `is_stable`
- `decomposition_products`

### Structural Properties
- `symmetry` (crystal system, space group, etc.)
- `equivalent_labels`, `wyckoff_symbols`
- `coordination_numbers`

### Mechanical Properties (when available)
- `elasticity` data
- `bulk_modulus`, `shear_modulus`
- `elastic_tensor`

### Magnetic Properties (when available)
- `magnetic_moments`
- `magnetic_ordering`
- `total_magnetization`

### And Many More...
- Dielectric properties
- Piezoelectric constants
- Thermal properties
- Surface properties
- Defect properties
- And other specialized calculated properties

## Performance Considerations

### File Sizes
- **Standard mode**: ~20-30 columns, moderate file sizes
- **All attributes mode**: 50-150+ columns, much larger files

### Download Times
- **Standard mode**: Faster downloads
- **All attributes mode**: Longer downloads due to more data

### Memory Usage
- **All attributes mode** uses more memory for processing
- Consider using `limit` parameter for initial testing

## Best Practices

### 1. Start Small
```python
# Test with a small sample first
df_test = downloader.download_all_perovskite_attributes(limit=10)
print(f"Available attributes: {list(df_test.columns)}")
```

### 2. Use Limits for Large Datasets
```python
# For production use, consider reasonable limits
df = downloader.download_all_perovskite_attributes(limit=1000)
```

### 3. Check Available Attributes
```python
# See what fields are available
available_fields = downloader.get_available_fields()
print(f"Total available fields: {len(available_fields)}")
```

### 4. Handle Large DataFrames
```python
# For very wide DataFrames, consider chunked processing
if len(df.columns) > 100:
    print(f"Large DataFrame with {len(df.columns)} columns")
    # Process in chunks or select specific columns for analysis
```

## Error Handling

The system gracefully handles:
- Missing fields in some materials
- Complex nested data structures
- Serialization issues with structure objects
- Memory limitations

Complex objects are automatically converted to strings for CSV compatibility.

## Files Created

When using all attributes mode, you'll get:
- `filename.csv`: Complete data with all attributes
- `filename.json`: Raw JSON data (when possible)
- `filename_summary.txt`: Statistics and metadata

The CSV files will have significantly more columns but remain compatible with standard data analysis tools.