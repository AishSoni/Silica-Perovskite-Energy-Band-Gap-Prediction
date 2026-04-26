# Limitations

## DFT Target Limitations

The models predict Materials Project DFT band gaps, not experimental band gaps. DFT band gaps can systematically underestimate experimental values.

## Dataset Scope

The corrected target scope is general double perovskites in the `ABC2D6` / `A2BB'X6` family. Formula pattern alone is not sufficient proof of structure, so the dataset must pass the validation gate before modeling.

## Regeneration Status

All previous generated results were removed from the active project. Until the corrected pipeline is rerun, no performance claim should be treated as final.

## Remaining Risks

- Materials Project database snapshots can change over time.
- Structural validation based on available summary fields may still need manual review for borderline compounds.
- The direct/indirect label depends on the underlying DFT calculation and k-point sampling.
- Experimental validation is still required before screening conclusions can be considered practical.

