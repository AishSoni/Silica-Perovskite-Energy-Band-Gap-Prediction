# Limitations and Future Work

## 1. DFT Limitations

### 1.1 Systematic Bandgap Underestimation
The most significant limitation stems from the training data itself. All bandgap values are computed using density functional theory (DFT) with the GGA-PBE functional, which is well-known to:

- **Underestimate bandgaps** by 30-50% compared to experimental values
- **Underestimate more severely** for larger bandgaps (>3 eV)
- **Predict some insulators as metals** (zero bandgap) due to underestimation

**Impact on this work:**
- Our ML models learn to reproduce DFT predictions, not experimental reality
- A model with MAE = 0.3 eV on DFT data may have MAE > 0.5 eV on experimental data
- Predicted "optimal" solar cell materials (Eg ~ 1.5 eV) may actually have Eg ~ 2-2.5 eV

**Potential solutions:**
- Retrain on hybrid functional (HSE06) or GW-corrected bandgaps (computationally expensive but more accurate)
- Apply empirical correction factors (e.g., multiply DFT bandgaps by 1.4)
- Train separate models on experimental bandgap data (limited availability)

### 1.2 Missing Electronic Structure Details
DFT provides only scalar bandgap values. It does not capture:

- **Bandgap type ambiguity:** Direct vs. indirect classification depends on k-point sampling density
- **Exciton binding energies:** Important for optoelectronic applications
- **Defect levels:** Intrinsic defects can create states in the gap
- **Temperature effects:** DFT is 0 K; real devices operate at 300 K

## 2. Dataset Limitations

### 2.1 Snapshot Dependency
- Downloaded from Materials Project on October 11, 2025
- Materials Project database continuously updated with new materials and revised calculations
- Different download dates yield different datasets
- Original paper (2024) used earlier snapshot → direct comparison difficult

### 2.2 Coverage Gaps
- Focused on ABC₂D₆ double perovskites only
- Does not include:
  - Single perovskites (ABX₃)
  - Halide perovskites (important for solar cells)
  - Layered or Ruddlesden-Popper phases
  - Experimentally synthesized but not computed materials

### 2.3 Missing Experimental Validation
- No comparison with measured bandgaps
- Cannot assess real-world prediction accuracy
- Synthesizability predictions based only on energy above hull (incomplete criterion)

## 3. Feature Engineering Limitations

### 3.1 Structural Information Underutilized
- Used only lattice parameters (a, b, c, α, β, γ)
- Did not employ:
  - **Graph neural networks** (GNN) on crystal structure graphs
  - **Crystal site descriptors** (local coordination environments)
  - **Voronoi tessellation features**
  - **Atom-pair distance distributions**

GNNs (e.g., CGCNN, MEGNet, SchNet) have shown superior performance by learning directly from atomic coordinates and periodic boundary conditions.

### 3.2 Missing HOMO-LUMO Features
The original paper used ionic HOMO/LUMO levels as features. We could not replicate this fully because:

- HOMO/LUMO data not directly available in Materials Project API
- Would require separate database lookup or quantum chemistry calculations
- May account for minor performance differences vs. original study

### 3.3 Feature Selection Not Performed
- Generated ~300 features but used all without selection
- Likely includes redundant or low-signal features
- Feature selection (e.g., LASSO, mutual information, recursive feature elimination) could:
  - Reduce overfitting
  - Improve interpretability
  - Speed up training

## 4. Modeling Limitations

### 4.1 Interpolation vs. Extrapolation
ML models perform well for:

- **Interpolation:** Compositions similar to training data
- **Extrapolation:** Performance degrades for novel chemical spaces

**Evidence:**
- Errors increase for rare element combinations
- Lower confidence for compositions with elements appearing <10 times in training set

**Recommendation:** Use model uncertainty estimates (e.g., Gaussian process regression, Bayesian neural networks, ensemble disagreement) before experimental validation.

### 4.2 No Uncertainty Quantification
- Point predictions only (no confidence intervals)
- Cannot distinguish:
  - High-confidence predictions in well-explored regions
  - Low-confidence extrapolations to novel chemistries

**Future improvement:**
- Implement conformal prediction, dropout-based uncertainty, or ensemble methods

### 4.3 Class Imbalance (Classification)
- Direct vs. indirect bandgap classification
- If one class is rare, SMOTE introduces synthetic samples
- Synthetic samples may not reflect true physics
- Model may overfit to interpolated synthetic data

## 5. Reproducibility Challenges

### 5.1 Random Seed Sensitivity
Despite setting seed=42:

- Some non-deterministic operations in XGBoost/LightGBM on certain hardware
- GPU-accelerated training introduces non-reproducibility
- Minor variations (<1%) in metrics across runs expected

### 5.2 Software Version Dependency
- Results depend on exact package versions
- API changes in matminer/pymatgen can alter feature values slightly
- Recommend containerization (Docker) for perfect reproducibility

## 6. Candidate Selection Limitations

### 6.1 Tolerance Factor Criterion
Goldschmidt tolerance factor (τ < 4.18) is a heuristic, not a guarantee of synthesis:

- Based on ionic radii (simplified model)
- Does not account for:
  - Covalent bonding character
  - Jahn-Teller distortions
  - Orbital overlap requirements

### 6.2 Energy Above Hull Incomplete
E_hull < 0.2 eV/atom suggests thermodynamic favorability, but:

- Does not guarantee kinetic accessibility
- Competing phases may form instead
- Synthesis conditions (temperature, pressure, atmosphere) matter

### 6.3 No Stability Testing
Predicted candidates not assessed for:

- Thermal stability (molecular dynamics)
- Moisture sensitivity
- Photostability (important for solar cells)
- Phase transitions

## 7. Computational Resource Constraints

- Featurization and training took [X] hours on [hardware]
- SHAP analysis computationally expensive (only subset of test set analyzed)
- Full hyperparameter search with Optuna not performed (used grid search for speed)
- Larger parameter grids or Bayesian optimization could improve performance further

## 8. Future Work

### 8.1 Immediate Extensions
1. **Retrain on higher-accuracy data:**
   - Use HSE06 or GW bandgaps if available
   - Incorporate experimental bandgaps from literature
   
2. **Expand to halide perovskites:**
   - ABX₃ formulations (e.g., CH₃NH₃PbI₃)
   - Important for solar cell applications

3. **Implement graph neural networks:**
   - CGCNN, MEGNet, SchNet for structure-aware learning
   - Potential 10-20% improvement in MAE

4. **Add uncertainty quantification:**
   - Gaussian process regression
   - Ensemble disagreement
   - Bayesian neural networks

5. **Active learning loop:**
   - Predict → synthesize → measure → retrain
   - Guide experimental discovery

### 8.2 Advanced Research Directions

1. **Multi-task learning:**
   - Predict bandgap, band edges, effective masses, dielectric constants simultaneously
   - Shared representations improve generalization

2. **Transfer learning:**
   - Pre-train on all Materials Project data
   - Fine-tune on perovskites

3. **Physics-informed ML:**
   - Incorporate k·p theory, tight-binding models
   - Constrain predictions to obey band theory

4. **Generative models:**
   - VAE or diffusion models to generate novel perovskite structures
   - Optimize in latent space for target properties

5. **Experimental validation:**
   - Collaborate with synthesis groups
   - Measure bandgaps of top predicted candidates
   - Assess model accuracy on real materials

### 8.3 Integration with DFT

**Closed-loop workflow:**
1. ML predicts promising candidates
2. DFT validates top-N candidates (expensive but accurate)
3. Retrain ML model on new DFT data
4. Repeat

This hybrid approach combines ML speed with DFT accuracy.

## 9. Lessons Learned

1. **Data quality matters more than model complexity:**
   - DFT underestimation is a bigger limitation than model choice
   - LightGBM vs. XGBoost differences < DFT error

2. **Feature engineering is critical:**
   - ~300 features from composition alone achieves good performance
   - Physics-informed features (electronegativity, radii) most important

3. **Interpretability is essential for materials science:**
   - SHAP analysis builds trust in predictions
   - Black-box models alone insufficient for scientific discovery

4. **Reproducibility requires diligence:**
   - Save seeds, versions, data snapshots
   - Document every decision (imputation, scaling, hyperparameters)

## 10. Recommendations for Future Users

1. **Before training:**
   - Verify data quality (check for outliers, errors in Materials Project)
   - Understand your target property distribution
   - Plan dataset splits carefully (stratified if imbalanced)

2. **During modeling:**
   - Start simple (linear regression) to establish baseline
   - Use cross-validation to assess generalization
   - Monitor training curves (detect overfitting early)
   - Try multiple imputation strategies (can impact results by 10-20%)

3. **After training:**
   - Interpret model decisions (SHAP, feature importance)
   - Test on out-of-distribution samples if available
   - Report uncertainty estimates alongside predictions
   - Validate experimentally before large-scale synthesis efforts

4. **For publication:**
   - Share code, data splits, and model checkpoints
   - Report all preprocessing steps
   - Compare with established baselines
   - Acknowledge limitations transparently

---

**Conclusion:**

While ML offers powerful tools for materials discovery, it is not a replacement for physics-based understanding or experimental validation. This work demonstrates that ML can effectively reproduce DFT predictions for perovskite bandgaps, but the next frontier is bridging the gap to experimental reality through improved training data, hybrid DFT-ML workflows, and systematic experimental validation.
