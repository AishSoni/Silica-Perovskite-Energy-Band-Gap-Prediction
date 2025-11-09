"""
Enhanced Materials Project API Data Downloader with Environment Configuration

This script uses a .env file to manage API keys and configuration settings securely.
"""

from config import config
from mp_api.client import MPRester
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import warnings


class EnhancedMaterialsProjectDownloader:
    """
    Enhanced Materials Project downloader with configuration management.
    """
    
    def __init__(self, use_document_model: bool = True):
        """Initialize the downloader using configuration from .env file.
        
        Args:
            use_document_model: If False, returns raw dictionaries instead of document objects.
                               Recommended for large downloads as per MP API documentation.
        """
        self.api_key = config.get_api_key()  # This will raise an error if not configured
        self.mpr = MPRester(self.api_key, use_document_model=use_document_model)
        self.use_document_model = use_document_model
        self.data_dir = config.OUTPUT_DIRECTORY
        self.data_dir.mkdir(exist_ok=True)
        self.max_results = config.MAX_RESULTS_PER_QUERY
        print(f"Initialized Materials Project downloader with API key: {self.api_key[:8]}...")
        print(f"Data will be saved to: {self.data_dir}")
        print(f"Using document model: {use_document_model}")
    
    def _validate_search_criteria(self, **criteria) -> Dict[str, Any]:
        """Validate and clean search criteria before sending to API."""
        validated = {}
        
        # Clean up None values
        for key, value in criteria.items():
            if value is not None:
                validated[key] = value
        
        # Validate element list length to avoid API character limit
        if 'elements' in validated and validated['elements']:
            element_str = ','.join(validated['elements'])
            if len(element_str) > 50:  # Conservative limit
                print(f"Warning: Element list may be too long ({len(element_str)} chars). Consider reducing elements.")
                # Optionally truncate the list
                validated['elements'] = validated['elements'][:8]  # Limit to first 8 elements
                print(f"Truncated to first 8 elements: {validated['elements']}")
        
        return validated
    
    def search_materials(
        self,
        elements: Optional[List[str]] = None,
        formula: Optional[str] = None,
        crystal_system: Optional[str] = None,
        spacegroup_number: Optional[int] = None,
        band_gap_range: Optional[tuple] = None,
        energy_above_hull_max: Optional[float] = None,
        theoretical: Optional[bool] = None,
        nsites_range: Optional[tuple] = None,
        volume_range: Optional[tuple] = None,
        fields: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[Any]:
        """
        Search for materials with enhanced defaults from configuration.
        """
        
        # Use config defaults if not specified
        if energy_above_hull_max is None:
            energy_above_hull_max = config.DEFAULT_ENERGY_ABOVE_HULL_MAX
        
        if band_gap_range is None:
            band_gap_range = config.DEFAULT_BAND_GAP_RANGE
        
        if nsites_range is None:
            nsites_range = config.DEFAULT_NSITES_RANGE  # Revert back to original name
        
        if limit is None:
            limit = self.max_results
        
        # Default fields to retrieve (using current API field names)
        if fields is None:
            fields = [
                "material_id", "formula_pretty", "structure", "energy_per_atom",
                "formation_energy_per_atom", "energy_above_hull", "band_gap",
                "density", "volume", "nsites", "symmetry", "theoretical", 
                "deprecated", "elements", "nelements", "composition_reduced"
            ]
        
        print(f"Searching for materials with criteria:")
        if elements:
            print(f"  - Elements: {elements}")
        if formula:
            print(f"  - Formula: {formula}")
        if crystal_system:
            print(f"  - Crystal system: {crystal_system}")
        if spacegroup_number:
            print(f"  - Space group: {spacegroup_number}")
        if band_gap_range:
            print(f"  - Band gap range: {band_gap_range[0]}-{band_gap_range[1]} eV")
        if nsites_range:
            print(f"  - Number of sites: {nsites_range[0]}-{nsites_range[1]}")
        print(f"  - Max energy above hull: {energy_above_hull_max} eV/atom")
        if limit and limit < self.max_results:
            print(f"  - Result limit: {limit}")
        else:
            print(f"  - Result limit: {self.max_results}")
        
        # Build search criteria dictionary
        search_criteria = self._validate_search_criteria(
            elements=elements,
            formula=formula,
            crystal_system=crystal_system,
            spacegroup_number=spacegroup_number,
            band_gap=band_gap_range,
            energy_above_hull=(None, energy_above_hull_max),
            theoretical=theoretical,
            nsites=nsites_range,  # Use nsites for search parameter
            volume=volume_range,
            fields=fields
        )
        
        try:
            # Perform the search with error handling
            print("Sending query to Materials Project API...")
            
            # Remove limit from search_criteria as it's not supported by search()
            search_params = search_criteria.copy()
            search_params.pop('limit', None)  # Remove limit if present
            
            docs = self.mpr.materials.summary.search(**search_params)
            
            # Limit results manually if needed
            if limit and len(docs) > limit:
                print(f"Found {len(docs)} materials, limiting to {limit} results")
                docs = docs[:limit]
            else:
                print(f"Found {len(docs)} materials matching criteria")
            
            if not docs:
                print("Warning: No materials found matching the specified criteria.")
                print("Consider relaxing your search parameters.")
            
            return docs
            
        except Exception as e:
            print(f"Error querying Materials Project API: {e}")
            print("This might be due to:")
            print("- Network connectivity issues")
            print("- API rate limiting")
            print("- Invalid search parameters")
            print("- API key issues")
            raise
    
    def search_by_element_groups(
        self,
        element_group: str,
        additional_elements: Optional[List[str]] = None,
        **kwargs
    ) -> List[Any]:
        """
        Search using predefined element groups from configuration.
        
        Args:
            element_group: One of 'transition_metals', 'lanthanides', 'semiconductors', 'oxides_common'
            additional_elements: Additional elements to include in search
            **kwargs: Additional search parameters
        """
        
        element_groups = {
            'transition_metals': config.TRANSITION_METALS,
            'lanthanides': config.LANTHANIDES,
            'semiconductors': config.SEMICONDUCTORS,
            'oxides_common': config.OXIDES_COMMON
        }
        
        if element_group not in element_groups:
            raise ValueError(f"Unknown element group: {element_group}. Available: {list(element_groups.keys())}")
        
        elements = element_groups[element_group].copy()
        if additional_elements:
            elements.extend(additional_elements)
        
        return self.search_materials(elements=elements, **kwargs)
    
    def get_available_fields(self) -> List[str]:
        """
        Get available fields from the Materials Project API.
        This helps ensure we're using correct field names.
        """
        try:
            available_fields = self.mpr.materials.summary.available_fields
            print(f"Available fields from MP API: {len(available_fields)} total")
            return available_fields
        except Exception as e:
            print(f"Warning: Could not retrieve available fields: {e}")
            # Return commonly known fields as fallback
            return [
                "material_id", "formula_pretty", "structure", "energy_per_atom",
                "formation_energy_per_atom", "energy_above_hull", "band_gap",
                "density", "volume", "nsites", "symmetry", "theoretical", 
                "deprecated", "elements", "nelements", "composition_reduced"
            ]
    
    def preview_search_results(self, docs: List[Any], max_preview: int = 5) -> None:
        """
        Preview search results before saving.
        Useful for validating data before full download.
        """
        if not docs:
            print("No results to preview")
            return
        
        print(f"\nPreview of first {min(len(docs), max_preview)} results:")
        print("=" * 60)
        
        for i, doc in enumerate(docs[:max_preview]):
            # Handle both document objects and raw dictionaries
            if isinstance(doc, dict):
                material_id = doc.get('material_id', 'Unknown')
                formula = doc.get('formula_pretty', 'Unknown')
                band_gap = doc.get('band_gap', 'N/A')
                energy_hull = doc.get('energy_above_hull', 'N/A')
            else:
                material_id = getattr(doc, 'material_id', 'Unknown')
                formula = getattr(doc, 'formula_pretty', 'Unknown')
                band_gap = getattr(doc, 'band_gap', 'N/A')
                energy_hull = getattr(doc, 'energy_above_hull', 'N/A')
            
            print(f"{i+1}. {material_id}: {formula}")
            print(f"   Band gap: {band_gap} eV, Energy above hull: {energy_hull} eV/atom")
        
        if len(docs) > max_preview:
            print(f"... and {len(docs) - max_preview} more results")
        print("=" * 60)
    
    def download_large_dataset(
        self,
        search_criteria: Dict[str, Any],
        filename_prefix: str = "large_dataset",
        use_raw_format: bool = True
    ) -> pd.DataFrame:
        """
        Download large datasets efficiently using API best practices.
        
        Args:
            search_criteria: Dictionary of search parameters
            filename_prefix: Prefix for saved files
            use_raw_format: Use raw dictionaries instead of document objects (faster for large datasets)
        
        Returns:
            DataFrame of the downloaded data
        """
        print("Starting large dataset download...")
        print("Using optimized settings for large downloads:")
        print(f"- Raw format (no document model): {use_raw_format}")
        print(f"- Search criteria: {search_criteria}")
        
        # Create a new MPRester instance optimized for large downloads
        if use_raw_format:
            mpr_large = MPRester(self.api_key, use_document_model=False, monty_decode=False)
        else:
            mpr_large = self.mpr
        
        try:
            # Remove limit from search criteria for the API call
            search_params = search_criteria.copy()
            search_params.pop('limit', None)
            
            docs = mpr_large.materials.summary.search(**search_params)
            
            if docs:
                # Apply limit manually if specified
                if 'limit' in search_criteria and search_criteria['limit']:
                    limit_val = search_criteria['limit']
                    if len(docs) > limit_val:
                        docs = docs[:limit_val]
                        print(f"Limited results to {limit_val} materials")
                
                print(f"Downloaded {len(docs)} materials")
                if use_raw_format:
                    # For raw format, we need to handle the data differently
                    df = self._convert_raw_to_dataframe(docs)
                else:
                    df = self.convert_to_dataframe(docs)
                
                # Save the data
                self._save_large_dataset(docs, df, filename_prefix, use_raw_format)
                return df
            else:
                print("No data found matching criteria")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error in large dataset download: {e}")
            raise
        finally:
            if use_raw_format and 'mpr_large' in locals():
                # Clean up the connection
                pass
    
    def _convert_raw_to_dataframe(self, raw_docs: List[Dict]) -> pd.DataFrame:
        """Convert raw dictionary documents to DataFrame."""
        if not raw_docs:
            return pd.DataFrame()
        
        # For raw documents, we can directly use pandas
        df = pd.DataFrame(raw_docs)
        
        # Clean up complex nested structures for CSV compatibility
        for col in df.columns:
            if col in ['structure', 'symmetry'] and col in df.columns:
                # Convert complex objects to strings for CSV compatibility
                df[col] = df[col].astype(str)
        
        print(f"Converted {len(raw_docs)} raw documents to DataFrame")
        return df
    
    def _save_large_dataset(self, docs: List[Any], df: pd.DataFrame, filename_prefix: str, is_raw: bool):
        """Save large datasets with appropriate handling."""
        if df.empty:
            print("No data to save")
            return
        
        # Always save CSV for large datasets
        csv_path = self.data_dir / f"{filename_prefix}.csv"
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved CSV data to: {csv_path}")
        
        # For large datasets, optionally skip JSON to save space and time
        if len(docs) < 10000:  # Only save JSON for smaller "large" datasets
            try:
                json_path = self.data_dir / f"{filename_prefix}.json"
                if is_raw:
                    # Raw docs are already JSON-serializable
                    with open(json_path, 'w') as f:
                        json.dump(docs, f, indent=2, default=str)
                else:
                    # Handle document objects
                    json_data = [doc.dict() if hasattr(doc, 'dict') else doc.__dict__ for doc in docs]
                    with open(json_path, 'w') as f:
                        json.dump(json_data, f, indent=2, default=str)
                print(f"✓ Saved JSON data to: {json_path}")
            except Exception as e:
                print(f"Warning: Could not save JSON (continuing with CSV): {e}")
        else:
            print("Skipping JSON save for large dataset (>10k materials)")
        
        # Save summary
        stats_path = self.data_dir / f"{filename_prefix}_summary.txt"
        with open(stats_path, 'w') as f:
            f.write(f"Large Dataset Summary\n")
            f.write(f"===================\n\n")
            f.write(f"Total materials: {len(docs)}\n")
            f.write(f"Date downloaded: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Raw format used: {is_raw}\n")
            f.write(f"CSV file: {csv_path.name}\n")
            f.write(f"Columns: {list(df.columns)}\n")
        
        print(f"✓ Saved summary to: {stats_path}")
    
    def convert_to_dataframe(self, docs: List[Any], all_attributes: bool = False) -> pd.DataFrame:
        """Convert list of SummaryDoc objects or raw dictionaries to a pandas DataFrame.
        
        Args:
            docs: List of material documents
            all_attributes: If True, attempts to preserve all available attributes in the DataFrame
        """
        if not docs:
            print("Warning: No documents to convert to DataFrame")
            return pd.DataFrame()
        
        data = []
        
        # If all_attributes is True, try to get all possible keys from the first few documents
        all_possible_keys = set()
        if all_attributes and docs:
            print("Scanning documents to identify all available attributes...")
            # Check first few documents to get all possible keys
            sample_size = min(10, len(docs))
            for doc in docs[:sample_size]:
                if isinstance(doc, dict):
                    all_possible_keys.update(doc.keys())
                else:
                    all_possible_keys.update(vars(doc).keys())
            
            print(f"Found {len(all_possible_keys)} unique attributes across sample documents")
            if len(all_possible_keys) > 50:
                print("Note: Large number of attributes detected - this will create a wide DataFrame")
        
        for doc in docs:
            row = {}
            
            # Handle both document objects and raw dictionaries
            if isinstance(doc, dict):
                # Raw dictionary (when use_document_model=False)
                get_val = lambda key, default=None: doc.get(key, default)
                available_keys = doc.keys() if all_attributes else []
            else:
                # Document object (when use_document_model=True)
                get_val = lambda key, default=None: getattr(doc, key, default)
                available_keys = vars(doc).keys() if all_attributes else []
            
            if all_attributes:
                # Include all available attributes
                for key in all_possible_keys:
                    value = get_val(key)
                    
                    # Handle complex nested objects
                    if value is not None:
                        if isinstance(value, (list, dict)) and key not in ['elements']:
                            # Convert complex objects to string for CSV compatibility
                            if key == 'structure':
                                # Special handling for structure objects
                                try:
                                    if hasattr(value, 'as_dict'):
                                        row[key] = str(value.as_dict())
                                    else:
                                        row[key] = str(value)
                                except:
                                    row[key] = str(value)
                            else:
                                row[key] = str(value)
                        elif key == 'elements' and isinstance(value, list):
                            # Special handling for elements list
                            row[key] = ','.join([str(el) for el in value])
                        else:
                            row[key] = value
                    else:
                        row[key] = None
            else:
                # Use the original selective attribute extraction
                # Basic properties - check if they exist before accessing
                row['material_id'] = get_val('material_id')
                row['formula_pretty'] = get_val('formula_pretty')
                row['energy_per_atom'] = get_val('energy_per_atom')
                row['formation_energy_per_atom'] = get_val('formation_energy_per_atom')
                row['energy_above_hull'] = get_val('energy_above_hull')
                row['band_gap'] = get_val('band_gap')
                row['density'] = get_val('density')
                row['volume'] = get_val('volume')
                
                # Handle both possible field names for number of sites
                nsites = get_val('nsites') or get_val('num_sites')
                row['nsites'] = nsites
                
                row['theoretical'] = get_val('theoretical')
                row['deprecated'] = get_val('deprecated')
                
                # Elements - check for both possible field names
                elements = get_val('elements')
                if elements:
                    # Convert elements to a string for CSV compatibility
                    row['elements'] = ','.join([str(el) for el in elements])
                    row['num_elements'] = len(elements)
                else:
                    row['elements'] = ''
                    row['num_elements'] = 0
                
                # Add nelements if available
                nelements = get_val('nelements')
                if nelements is not None:
                    row['num_elements'] = nelements
                
                # Symmetry info (from symmetry field)
                symmetry = get_val('symmetry')
                if symmetry:
                    if isinstance(symmetry, dict):
                        row['crystal_system'] = symmetry.get('crystal_system')
                        row['spacegroup_number'] = symmetry.get('number')
                        row['spacegroup_symbol'] = symmetry.get('symbol')
                    else:
                        row['crystal_system'] = getattr(symmetry, 'crystal_system', None)
                        row['spacegroup_number'] = getattr(symmetry, 'number', None)
                        row['spacegroup_symbol'] = getattr(symmetry, 'symbol', None)
                else:
                    row['crystal_system'] = None
                    row['spacegroup_number'] = None
                    row['spacegroup_symbol'] = None
                
                # Structure information (summary) - simplified for CSV
                structure = get_val('structure')
                if structure:
                    try:
                        if isinstance(structure, dict):
                            lattice = structure.get('lattice', {})
                            row['lattice_a'] = lattice.get('a')
                            row['lattice_b'] = lattice.get('b')
                            row['lattice_c'] = lattice.get('c')
                            row['lattice_alpha'] = lattice.get('alpha')
                            row['lattice_beta'] = lattice.get('beta')
                            row['lattice_gamma'] = lattice.get('gamma')
                        elif hasattr(structure, 'lattice'):
                            lattice = structure.lattice
                            row['lattice_a'] = getattr(lattice, 'a', None)
                            row['lattice_b'] = getattr(lattice, 'b', None)
                            row['lattice_c'] = getattr(lattice, 'c', None)
                            row['lattice_alpha'] = getattr(lattice, 'alpha', None)
                            row['lattice_beta'] = getattr(lattice, 'beta', None)
                            row['lattice_gamma'] = getattr(lattice, 'gamma', None)
                        else:
                            # Handle other structure representations
                            for key in ['lattice_a', 'lattice_b', 'lattice_c', 'lattice_alpha', 'lattice_beta', 'lattice_gamma']:
                                row[key] = None
                    except Exception as e:
                        print(f"Warning: Could not extract lattice parameters: {e}")
                        for key in ['lattice_a', 'lattice_b', 'lattice_c', 'lattice_alpha', 'lattice_beta', 'lattice_gamma']:
                            row[key] = None
                else:
                    for key in ['lattice_a', 'lattice_b', 'lattice_c', 'lattice_alpha', 'lattice_beta', 'lattice_gamma']:
                        row[key] = None
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        if all_attributes:
            print(f"Converted {len(data)} materials to DataFrame with {len(df.columns)} columns (all attributes mode)")
        else:
            print(f"Converted {len(data)} materials to DataFrame with {len(df.columns)} columns")
        
        return df
    
    def save_data(self, docs: List[Any], filename_prefix: str = "materials", all_attributes: bool = False):
        """Save materials data in multiple formats.
        
        Args:
            docs: List of material documents
            filename_prefix: Prefix for output files
            all_attributes: If True, attempts to preserve all available attributes
        """
        if not docs:
            print("Warning: No data to save")
            return None
        
        try:
            df = self.convert_to_dataframe(docs, all_attributes=all_attributes)
            
            if df.empty:
                print("Warning: Empty DataFrame, no data to save")
                return None
            
            # Save as CSV
            csv_path = self.data_dir / f"{filename_prefix}.csv"
            df.to_csv(csv_path, index=False)
            print(f"✓ Saved CSV data to: {csv_path}")
            
            if all_attributes:
                print(f"Note: CSV contains {len(df.columns)} columns with all available attributes")
            
            # Save as JSON (with structures) - handle serialization carefully
            json_data = []
            for doc in docs:
                try:
                    if isinstance(doc, dict):
                        doc_dict = doc.copy()
                    else:
                        doc_dict = doc.dict() if hasattr(doc, 'dict') else doc.__dict__
                    
                    # Convert Structure objects to dict for JSON serialization
                    if 'structure' in doc_dict and doc_dict['structure']:
                        if hasattr(doc_dict['structure'], 'as_dict'):
                            doc_dict['structure'] = doc_dict['structure'].as_dict()
                        elif not isinstance(doc_dict['structure'], dict):
                            doc_dict['structure'] = str(doc_dict['structure'])
                    json_data.append(doc_dict)
                except Exception as e:
                    material_id = 'unknown'
                    if isinstance(doc, dict):
                        material_id = doc.get('material_id', 'unknown')
                    else:
                        material_id = getattr(doc, 'material_id', 'unknown')
                    print(f"Warning: Could not serialize document {material_id}: {e}")
                    continue
            
            if json_data:
                json_path = self.data_dir / f"{filename_prefix}.json"
                with open(json_path, 'w') as f:
                    json.dump(json_data, f, indent=2, default=str)
                print(f"✓ Saved JSON data to: {json_path}")
            
            # Save summary statistics
            stats_path = self.data_dir / f"{filename_prefix}_summary.txt"
            with open(stats_path, 'w') as f:
                f.write(f"Materials Data Summary\n")
                f.write(f"=====================\n\n")
                f.write(f"Total materials: {len(docs)}\n")
                f.write(f"Date downloaded: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Only include statistics for columns that exist and have data
                if 'crystal_system' in df.columns and not df['crystal_system'].isna().all():
                    f.write(f"Crystal systems: {df['crystal_system'].value_counts().to_dict()}\n")
                
                if 'theoretical' in df.columns and not df['theoretical'].isna().all():
                    f.write(f"Theoretical vs Experimental: {df['theoretical'].value_counts().to_dict()}\n")
                
                if 'band_gap' in df.columns and not df['band_gap'].isna().all():
                    bg_min, bg_max = df['band_gap'].min(), df['band_gap'].max()
                    f.write(f"Band gap range: {bg_min:.3f} - {bg_max:.3f} eV\n")
                
                if 'energy_above_hull' in df.columns and not df['energy_above_hull'].isna().all():
                    eh_min, eh_max = df['energy_above_hull'].min(), df['energy_above_hull'].max()
                    f.write(f"Energy above hull range: {eh_min:.6f} - {eh_max:.6f} eV/atom\n")
                
                if 'nsites' in df.columns and not df['nsites'].isna().all():
                    ns_min, ns_max = df['nsites'].min(), df['nsites'].max()
                    f.write(f"Number of sites range: {int(ns_min)} - {int(ns_max)}\n")
                elif 'num_sites' in df.columns and not df['num_sites'].isna().all():
                    ns_min, ns_max = df['num_sites'].min(), df['num_sites'].max()
                    f.write(f"Number of sites range: {int(ns_min)} - {int(ns_max)}\n")
                
                if 'num_elements' in df.columns and not df['num_elements'].isna().all():
                    f.write(f"Number of elements distribution: {df['num_elements'].value_counts().to_dict()}\n")
            
            print(f"✓ Saved summary to: {stats_path}")
            return df
            
        except Exception as e:
            print(f"Error saving data: {e}")
            raise


    def query_all_perovskite_materials(self, all_attributes: bool = False) -> List[Any]:
        """
        Query ALL available Perovskite materials with specified or all available fields.
        
        Args:
            all_attributes: If True, downloads ALL available attributes by omitting fields parameter.
                           If False, downloads only the specified comprehensive fields.
        
        The default requested fields are:
        - material_id, formula_pretty, elements, nelements
        - structure, lattice, volume, density
        - crystal_system, symmetry.symbol, symmetry.number
        - band_gap, is_gap_direct
        - formation_energy_per_atom, energy_above_hull, is_stable
        - elasticity
        
        Returns:
            List of all Perovskite materials with requested properties
        """
        
        if all_attributes:
            # When all_attributes=True, we omit the fields parameter to get ALL available data
            fields = None
            print("Querying ALL Perovskite materials with ALL AVAILABLE ATTRIBUTES...")
            print("This will download all data fields available in the Materials Project API")
            print("WARNING: This may result in very large files and longer download times")
        else:
            # Exact fields as requested by the user (using only valid API field names)
            fields = [
                "material_id",
                "formula_pretty",
                "elements",
                "nelements", 
                "structure",  # This includes lattice information
                "volume",
                "density",
                "symmetry",  # This includes crystal_system, symbol, and number
                "band_gap",
                "is_gap_direct",
                "formation_energy_per_atom",
                "energy_above_hull",
                "is_stable",
                # Additional fields that were missing:
                "energy_per_atom",
                "nsites",
                "theoretical",
                "deprecated",
                # Note: 'lattice', 'crystal_system', and 'elasticity' are not directly available
                # but we can extract lattice from structure and crystal_system from symmetry
            ]
            
            print("Querying ALL Perovskite materials from Materials Project database...")
            print(f"Requesting {len(fields)} core fields per material:")
            for i, field in enumerate(fields, 1):
                print(f"  {i:2d}. {field}")
            print("Note: lattice data extracted from structure, crystal_system from symmetry")
            print("Note: elasticity data not available in summary endpoint")
        
        print()
        
        all_perovskite_materials = []
        
        # Strategy: Search for materials with known Perovskite compositions
        # Common Perovskite formulas and patterns
        perovskite_searches = [
            # Classic oxide Perovskites
            {'formula': '*TiO3'},
            {'formula': '*ZrO3'}, 
            {'formula': '*SnO3'},
            {'formula': '*PbO3'},
            {'formula': '*MnO3'},
            {'formula': '*FeO3'},
            {'formula': '*CoO3'},
            {'formula': '*NiO3'},
            {'formula': '*CrO3'},
            {'formula': '*VO3'},
            
            # Specific well-known Perovskites
            {'formula': 'BaTiO3'},
            {'formula': 'SrTiO3'},
            {'formula': 'CaTiO3'},
            {'formula': 'PbTiO3'},
            {'formula': 'LaFeO3'},
            {'formula': 'LaMnO3'},
            {'formula': 'LaCoO3'},
            {'formula': 'LaNiO3'},
            {'formula': 'BiFeO3'},
            
            # Halide Perovskites
            {'formula': '*PbI3'},
            {'formula': '*PbBr3'},
            {'formula': '*PbCl3'},
            {'formula': '*SnI3'},
            {'formula': '*GeI3'},
        ]
        
        print("Searching through multiple Perovskite composition patterns...")
        
        try:
            # Search using various strategies to capture all Perovskites
            for i, search_params in enumerate(perovskite_searches):
                try:
                    print(f"Search {i+1}/{len(perovskite_searches)}: {search_params}")
                    
                    # Build search arguments - only include fields if specified
                    search_args = search_params.copy()
                    if fields is not None:
                        search_args['fields'] = fields
                    
                    docs = self.mpr.materials.summary.search(**search_args)
                    
                    if docs:
                        print(f"  Found {len(docs)} materials")
                        all_perovskite_materials.extend(docs)
                    else:
                        print(f"  No materials found")
                        
                except Exception as e:
                    print(f"  Warning: Search failed: {e}")
                    continue
            
            # Additional broad search for potential Perovskites
            print("\nPerforming broad search for additional Perovskite structures...")
            
            # Search for materials with cubic crystal system and common Perovskite elements
            try:
                # Build search arguments - only include fields if specified
                broad_search_args = {
                    'crystal_system': "cubic",
                    'elements': ["O"],  # Most Perovskites contain oxygen
                    'energy_above_hull': (None, 0.2),  # Include slightly metastable phases
                }
                if fields is not None:
                    broad_search_args['fields'] = fields
                
                broad_docs = self.mpr.materials.summary.search(**broad_search_args)
                
                print(f"Broad cubic oxide search found {len(broad_docs)} materials")
                
                # Filter for Perovskite-like compositions
                perovskite_candidates = []
                for doc in broad_docs:
                    if self._is_likely_perovskite(doc):
                        perovskite_candidates.append(doc)
                
                print(f"Identified {len(perovskite_candidates)} additional Perovskite candidates")
                all_perovskite_materials.extend(perovskite_candidates)
                
            except Exception as e:
                print(f"Broad search failed: {e}")
            
            # Remove duplicates based on material_id
            unique_materials = {}
            for material in all_perovskite_materials:
                if isinstance(material, dict):
                    mat_id = material.get('material_id')
                else:
                    mat_id = getattr(material, 'material_id', None)
                
                if mat_id and mat_id not in unique_materials:
                    unique_materials[mat_id] = material
            
            final_materials = list(unique_materials.values())
            
            print(f"\nCOMPLETE: Found {len(final_materials)} unique Perovskite materials")
            print("Materials include both experimental and theoretical structures")
            print("Data includes all requested fields where available")
            
            return final_materials
            
        except Exception as e:
            print(f"Error in comprehensive Perovskite search: {e}")
            raise
    
    def _is_likely_perovskite(self, doc: Any) -> bool:
        """
        Enhanced check for Perovskite-like materials based on composition and structure.
        
        Args:
            doc: Material document
            
        Returns:
            True if material is likely a Perovskite
        """
        try:
            # Get material properties
            if isinstance(doc, dict):
                elements = doc.get('elements', [])
                formula = doc.get('formula_pretty', '')
                nelements = doc.get('nelements', 0)
                crystal_system = doc.get('crystal_system', '')
            else:
                elements = getattr(doc, 'elements', [])
                formula = getattr(doc, 'formula_pretty', '')
                nelements = getattr(doc, 'nelements', 0)
                crystal_system = getattr(doc, 'crystal_system', '')
            
            if not elements or not formula:
                return False
            
            # Perovskites typically have 3-4 elements (ABX3 or A2BB'X6, etc.)
            if nelements < 3 or nelements > 5:
                return False
            
            # Common Perovskite crystal systems
            perovskite_crystal_systems = ['cubic', 'tetragonal', 'orthorhombic', 'hexagonal', 'rhombohedral']
            if crystal_system.lower() not in perovskite_crystal_systems:
                return False
            
            # Check for Perovskite formula patterns
            perovskite_patterns = [
                'O3', 'F3', 'Cl3', 'Br3', 'I3',  # X3 endings
                'TiO3', 'ZrO3', 'SnO3', 'PbO3', 'MnO3', 'FeO3', 'CoO3', 'NiO3',  # Common BO3
                'CaTi', 'SrTi', 'BaTi', 'PbTi', 'LaFe', 'LaMn', 'LaCo', 'LaNi',  # Common AB combinations
            ]
            
            has_perovskite_pattern = any(pattern in formula for pattern in perovskite_patterns)
            
            # Check elemental composition for typical Perovskite elements
            a_site_elements = ['Ca', 'Sr', 'Ba', 'Pb', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'K', 'Rb', 'Cs', 'Bi', 'Na']
            b_site_elements = ['Ti', 'Zr', 'Hf', 'V', 'Nb', 'Ta', 'Cr', 'Mo', 'W', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Sn', 'Pb', 'Al', 'Ga', 'In']
            x_site_elements = ['O', 'F', 'Cl', 'Br', 'I', 'N', 'S']
            
            has_a_site = any(elem in a_site_elements for elem in elements)
            has_b_site = any(elem in b_site_elements for elem in elements)
            has_x_site = any(elem in x_site_elements for elem in elements)
            
            # Must have typical A, B, and X site elements
            has_typical_composition = has_a_site and has_b_site and has_x_site
            
            return has_perovskite_pattern or has_typical_composition
            
        except Exception as e:
            print(f"Warning: Error in Perovskite analysis: {e}")
            return False


    def download_all_perovskite_attributes(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Convenience method to download ALL available attributes for Perovskite materials.
        
        Args:
            limit: Maximum number of materials to download (None for all)
            
        Returns:
            DataFrame with all available attributes
            
        Warning:
            This can result in very large files and long download times.
            Consider using a limit for initial testing.
        """
        print("="*60)
        print("DOWNLOADING ALL PEROVSKITE ATTRIBUTES")
        print("="*60)
        print("WARNING: This will download ALL available data fields!")
        print("This may take significant time and create large files.")
        
        if limit:
            print(f"Limiting download to {limit} materials for testing.")
        else:
            print("No limit set - downloading ALL Perovskite materials.")
        
        print()
        
        # Get all perovskite materials with all attributes
        all_docs = self.query_all_perovskite_materials(all_attributes=True)
        
        if limit and len(all_docs) > limit:
            print(f"Limiting results from {len(all_docs)} to {limit} materials")
            all_docs = all_docs[:limit]
        
        if all_docs:
            print(f"Saving {len(all_docs)} Perovskite materials with all attributes...")
            
            # Save with all attributes
            df = self.save_data(
                all_docs, 
                "all_perovskite_complete_attributes",
                all_attributes=True
            )
            
            print(f"✓ Complete attribute download finished!")
            print(f"✓ {len(all_docs)} materials saved with {len(df.columns)} total attributes")
            
            return df
        else:
            print("No Perovskite materials found")
            return pd.DataFrame()


def main():
    """Main function with example usage and demonstration of all attributes download."""
    
    try:
        # Print current configuration
        config.print_config()
        print()
        
        # Initialize downloader (will validate API key)
        downloader = EnhancedMaterialsProjectDownloader()
        
        # Show available fields for reference
        print("Available API fields:")
        available_fields = downloader.get_available_fields()
        print(f"Total available fields: {len(available_fields)}")
        print()
        
        # Demonstrate both download modes
        print("="*80)
        print("MATERIALS PROJECT PEROVSKITE DOWNLOADER")
        print("="*80)
        print("This script demonstrates two download modes:")
        print("1. STANDARD MODE: Downloads key materials properties (recommended)")
        print("2. ALL ATTRIBUTES MODE: Downloads every available data field (comprehensive)")
        print("="*80)
        print()
        
        # Standard download demonstration
        print("DEMONSTRATION 1: Standard Perovskite Download")
        print("="*60)
        
        try:
            # Ask user which mode they prefer (for demonstration)
            print("PEROVSKITE DOWNLOAD OPTIONS:")
            print("1. Standard fields (faster, smaller files)")
            print("2. All available attributes (comprehensive but larger files)")
            print("Defaulting to standard fields for this demonstration...")
            print()
            
            # Query all Perovskite materials with standard fields
            print("="*60)
            print("DOWNLOADING WITH STANDARD FIELDS...")
            print("="*60)
            perovskite_docs_standard = downloader.query_all_perovskite_materials(all_attributes=False)
            
            if perovskite_docs_standard:
                # Preview results before saving
                downloader.preview_search_results(perovskite_docs_standard, max_preview=10)
                df_standard = downloader.save_data(
                    perovskite_docs_standard, 
                    "all_perovskite_materials_standard",
                    all_attributes=False
                )
                print(f"Downloaded {len(perovskite_docs_standard)} Perovskite materials with standard fields\n")
                
                # Optionally demonstrate all attributes download with a smaller subset
                print("="*60)
                print("DEMONSTRATING ALL ATTRIBUTES DOWNLOAD (first 10 materials)...")
                print("="*60)
                
                # Take only first 10 materials for all-attributes demo to avoid huge files
                sample_docs = perovskite_docs_standard[:10] if len(perovskite_docs_standard) >= 10 else perovskite_docs_standard
                
                # Re-query the sample with all attributes
                if sample_docs:
                    sample_material_ids = []
                    for doc in sample_docs:
                        if isinstance(doc, dict):
                            sample_material_ids.append(doc.get('material_id'))
                        else:
                            sample_material_ids.append(getattr(doc, 'material_id', None))
                    
                    # Query these specific materials with all attributes
                    try:
                        all_attrs_docs = []
                        for mat_id in sample_material_ids[:5]:  # Limit to 5 for demo
                            if mat_id:
                                try:
                                    doc_full = downloader.mpr.materials.summary.search(material_ids=[mat_id])
                                    if doc_full:
                                        all_attrs_docs.extend(doc_full)
                                except Exception as e:
                                    print(f"Warning: Could not get all attributes for {mat_id}: {e}")
                        
                        if all_attrs_docs:
                            df_all_attrs = downloader.save_data(
                                all_attrs_docs,
                                "perovskite_all_attributes_sample",
                                all_attributes=True
                            )
                            print(f"Sample download with ALL attributes: {len(all_attrs_docs)} materials")
                            print(f"Standard fields: {len(df_standard.columns)} columns")
                            print(f"All attributes: {len(df_all_attrs.columns)} columns")
                        
                    except Exception as e:
                        print(f"All attributes demo failed: {e}")
                
                # Use the standard dataset for analysis
                perovskite_docs = perovskite_docs_standard
                df = df_standard
                
                # Print detailed analysis of the Perovskite data
                print("PEROVSKITE MATERIALS ANALYSIS:")
                print(f"  - Total materials found: {len(perovskite_docs)}")
                
                if len(perovskite_docs) > 0:
                    # Analyze crystal systems
                    crystal_systems = {}
                    band_gaps = []
                    formation_energies = []
                    stable_materials = 0
                    
                    for doc in perovskite_docs:
                        # Get properties from document
                        if isinstance(doc, dict):
                            symmetry = doc.get('symmetry', {})
                            band_gap = doc.get('band_gap')
                            formation_energy = doc.get('formation_energy_per_atom')
                            is_stable = doc.get('is_stable', False)
                            energy_hull = doc.get('energy_above_hull')
                        else:
                            symmetry = getattr(doc, 'symmetry', {})
                            band_gap = getattr(doc, 'band_gap', None)
                            formation_energy = getattr(doc, 'formation_energy_per_atom', None)
                            is_stable = getattr(doc, 'is_stable', False)
                            energy_hull = getattr(doc, 'energy_above_hull', None)
                        
                        # Crystal system analysis
                        if isinstance(symmetry, dict):
                            crystal_sys = symmetry.get('crystal_system', 'Unknown')
                        else:
                            crystal_sys = getattr(symmetry, 'crystal_system', 'Unknown')
                        
                        crystal_systems[crystal_sys] = crystal_systems.get(crystal_sys, 0) + 1
                        
                        # Property analysis
                        if band_gap is not None:
                            band_gaps.append(band_gap)
                        if formation_energy is not None:
                            formation_energies.append(formation_energy)
                        if is_stable or (energy_hull is not None and energy_hull <= 0.05):
                            stable_materials += 1
                    
                    print(f"  - Crystal systems: {crystal_systems}")
                    if band_gaps:
                        print(f"  - Band gap range: {min(band_gaps):.3f} - {max(band_gaps):.3f} eV")
                        print(f"  - Average band gap: {sum(band_gaps)/len(band_gaps):.3f} eV")
                        print(f"  - Metallic materials (gap = 0): {sum(1 for bg in band_gaps if bg == 0)}")
                        print(f"  - Semiconducting (0 < gap < 3 eV): {sum(1 for bg in band_gaps if 0 < bg < 3)}")
                        print(f"  - Insulating (gap ≥ 3 eV): {sum(1 for bg in band_gaps if bg >= 3)}")
                    if formation_energies:
                        print(f"  - Formation energy range: {min(formation_energies):.3f} - {max(formation_energies):.3f} eV/atom")
                    print(f"  - Thermodynamically stable materials: {stable_materials}")
                    
                print(f"\n✓ All Perovskite data saved to: all_perovskite_materials.csv")
            else:
                print("No Perovskite materials found with the current criteria")
                
        except Exception as e:
            print(f"Error in Perovskite search: {e}")
            import traceback
            traceback.print_exc()
        
        print("="*60)
        print("Download complete!")
        print("="*60)
        print(f"Data saved to: {downloader.data_dir}")
        print("Files created:")
        for file_path in downloader.data_dir.glob("*"):
            if file_path.is_file():
                print(f"  - {file_path.name}")
        
        print("\n" + "="*80)
        print("HOW TO USE ALL ATTRIBUTES DOWNLOAD")
        print("="*80)
        print("To download ALL available attributes for Perovskite materials:")
        print()
        print("# Method 1: Using the convenience function")
        print("df_all = downloader.download_all_perovskite_attributes(limit=50)  # Limit for testing")
        print("df_complete = downloader.download_all_perovskite_attributes()  # No limit - all materials")
        print()
        print("# Method 2: Using the main query function")
        print("docs = downloader.query_all_perovskite_materials(all_attributes=True)")
        print("df = downloader.save_data(docs, 'my_perovskites', all_attributes=True)")
        print()
        print("# Method 3: For any material search with all attributes")
        print("docs = downloader.search_materials(elements=['Ti', 'O'])  # No fields parameter = all fields")
        print("df = downloader.save_data(docs, 'titanium_oxides', all_attributes=True)")
        print()
        print("IMPORTANT NOTES:")
        print("- All attributes mode downloads EVERY available field from the MP database")
        print("- This includes structure data, electronic properties, mechanical data, etc.")
        print("- Files will be significantly larger and download times longer")
        print("- Consider using a limit parameter for initial testing")
        print("- The resulting CSV files may have 100+ columns depending on available data")
        
        print("\n" + "="*60)
        print("Example: How to search for materials with specific properties")
        print("="*60)
        
        # Example of searching for materials with dielectric data (simplified)
        try:
            print("Searching for materials with dielectric data...")
            dielectric_docs = downloader.search_materials(
                elements=['Si'],  # Simple search
                fields=["material_id", "formula_pretty"],
                limit=5
            )
            if dielectric_docs:
                print(f"Found {len(dielectric_docs)} silicon-containing materials:")
                for doc in dielectric_docs[:3]:
                    print(f"  - {doc.material_id}: {doc.formula_pretty}")
        except Exception as e:
            print(f"Note: Could not demonstrate materials search: {e}")
        
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("\nPlease check your .env file and ensure MP_API_KEY is set correctly.")
        print("Visit https://next-gen.materialsproject.org/dashboard to get your API key.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        print("Full error details:")
        traceback.print_exc()


if __name__ == "__main__":
    main()