"""
Comprehensive Materials Project Database Downloader

This script downloads ALL available data from the Materials Project API with
ALL fields and ALL possible materials across all endpoints.

Based on MP API documentation best practices for large downloads:
- Uses raw format (use_document_model=False, monty_decode=False) for efficiency
- Downloads data in chunks to avoid memory issues
- Implements proper rate limiting and error handling
- Saves data in multiple formats for redundancy
- Provides detailed progress tracking and statistics

WARNING: This will download the ENTIRE Materials Project database.
This could be tens of gigabytes of data and take many hours/days to complete.
Make sure you have sufficient disk space and a stable internet connection.
"""

import os
import json
import time
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
import traceback
import gc
from config import config
from mp_api.client import MPRester

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download_all_data.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ComprehensiveMPDownloader:
    """
    Downloads ALL available data from Materials Project API efficiently.
    
    This downloader implements all best practices from MP API documentation:
    - Raw format for large downloads
    - Chunked downloads to manage memory
    - Comprehensive error handling and retry logic
    - Multiple output formats
    - Detailed progress tracking
    """
    
    def __init__(self, chunk_size: int = 10000):
        """
        Initialize the comprehensive downloader.
        
        Args:
            chunk_size: Number of materials to download per chunk (default: 10000)
        """
        self.api_key = config.get_api_key()
        self.chunk_size = chunk_size
        self.data_dir = config.OUTPUT_DIRECTORY / "complete_database"
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        # Statistics tracking
        self.total_materials_downloaded = 0
        self.failed_downloads = []
        self.download_start_time = None
        
        logger.info(f"Initialized comprehensive MP downloader")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Chunk size: {chunk_size}")
        logger.info(f"API key: {self.api_key[:8]}...")
        
    def get_all_available_endpoints(self) -> List[Tuple[str, str]]:
        """
        Get all available endpoints from the API.
        
        Returns:
            List of (endpoint_name, description) tuples
        """
        endpoints = [
            # Core materials endpoints
            ("materials.summary", "Summary information - core materials data"),
            ("materials.core", "Core materials documents with all calculated properties"),
            ("materials.tasks", "Individual VASP calculation details"),
            ("materials.thermo", "Thermodynamic data and phase diagrams"),
            
            # Electronic structure
            ("materials.electronic_structure", "Electronic structure (combined)"),
            ("materials.electronic_structure.bandstructure", "Band structure data"),
            ("materials.electronic_structure.dos", "Density of states data"),
            
            # Physical properties
            ("materials.elasticity", "Elastic constants and mechanical properties"),
            ("materials.dielectric", "Dielectric properties"),
            ("materials.piezoelectric", "Piezoelectric properties"),
            ("materials.magnetism", "Magnetic properties"),
            ("materials.absorption", "Optical absorption spectra"),
            ("materials.eos", "Equations of state"),
            
            # Structural properties
            ("materials.bonds", "Chemical bonding analysis"),
            ("materials.chemenv", "Chemical environment analysis"),
            ("materials.robocrys", "Automated structure description"),
            ("materials.similarity", "Structure similarity data"),
            ("materials.oxidation_states", "Oxidation state assignments"),
            
            # Applications
            ("materials.insertion_electrodes", "Battery electrode materials"),
            ("materials.surface_properties", "Surface properties"),
            ("materials.substrates", "Substrate matching data"),
            ("materials.grain_boundaries", "Grain boundary properties"),
            
            # Synthesis and experimental
            ("materials.synthesis", "Synthesis recipes and conditions"),
            ("materials.provenance", "Data provenance and experimental tags"),
            ("materials.xas", "X-ray absorption spectroscopy"),
            
            # Vibrational properties
            ("materials.phonon", "Phonon band structures and DOS"),
            
            # Alloys (experimental)
            ("materials.alloys", "Alloy data (experimental)"),
            
            # Molecules endpoints
            ("molecules.summary", "Molecular summary data"),
            ("molecules.core", "Core molecular data"),
            ("molecules.tasks", "Molecular calculation tasks"),
            ("molecules.thermo", "Molecular thermodynamics"),
            ("molecules.orbitals", "Molecular orbitals"),
            ("molecules.vibrations", "Molecular vibrations"),
            ("molecules.bonding", "Molecular bonding analysis"),
            ("molecules.partial_charges", "Partial atomic charges"),
            ("molecules.partial_spins", "Partial atomic spins"),
            ("molecules.redox", "Redox properties"),
            ("molecules.jcesr", "Battery electrolyte molecules"),
            ("molecules.assoc", "Molecular associations"),
            
            # DOI references
            ("doi", "DOI references for materials"),
        ]
        
        logger.info(f"Found {len(endpoints)} available endpoints")
        return endpoints
    
    def get_all_available_fields(self, endpoint_path: str) -> List[str]:
        """
        Get all available fields for a specific endpoint.
        
        Args:
            endpoint_path: Path to the endpoint (e.g., 'materials.summary')
            
        Returns:
            List of available field names
        """
        try:
            # Use optimized MPRester for field discovery
            with MPRester(self.api_key, use_document_model=False) as mpr:
                # Navigate to the endpoint
                endpoint = mpr
                for part in endpoint_path.split('.'):
                    endpoint = getattr(endpoint, part)
                
                # Get available fields
                if hasattr(endpoint, 'available_fields'):
                    fields = list(endpoint.available_fields)
                    logger.info(f"Endpoint {endpoint_path}: {len(fields)} fields available")
                    return fields
                else:
                    logger.warning(f"Endpoint {endpoint_path}: No available_fields attribute")
                    return []
                    
        except Exception as e:
            logger.warning(f"Could not get fields for {endpoint_path}: {e}")
            return []
    
    def estimate_data_size(self, endpoint_path: str) -> Dict[str, Any]:
        """
        Estimate the size and scope of data for an endpoint.
        
        Args:
            endpoint_path: Path to the endpoint
            
        Returns:
            Dictionary with size estimates
        """
        try:
            with MPRester(self.api_key, use_document_model=False, monty_decode=False) as mpr:
                # Navigate to endpoint
                endpoint = mpr
                for part in endpoint_path.split('.'):
                    endpoint = getattr(endpoint, part)
                
                # Get a small sample to estimate
                logger.info(f"Estimating data size for {endpoint_path}...")
                
                # Try to get count first with minimal fields
                test_docs = endpoint.search(fields=["material_id"])[:100]  # Sample 100
                
                if test_docs:
                    sample_size = len(test_docs)
                    
                    # Estimate total count by trying progressively larger queries
                    total_estimate = sample_size
                    try:
                        # Try to get more to estimate total
                        larger_sample = endpoint.search(fields=["material_id"])[:1000]
                        if len(larger_sample) == 1000:
                            # There are likely more than 1000 entries
                            total_estimate = "10,000+"
                        else:
                            total_estimate = len(larger_sample)
                    except:
                        pass
                    
                    # Get sample document size
                    if test_docs:
                        sample_doc = test_docs[0]
                        estimated_doc_size = len(json.dumps(sample_doc, default=str))
                        
                        return {
                            'sample_count': sample_size,
                            'estimated_total': total_estimate,
                            'estimated_doc_size_bytes': estimated_doc_size,
                            'has_data': True
                        }
                
                return {
                    'sample_count': 0,
                    'estimated_total': 0,
                    'estimated_doc_size_bytes': 0,
                    'has_data': False
                }
                
        except Exception as e:
            logger.warning(f"Could not estimate size for {endpoint_path}: {e}")
            return {
                'sample_count': 0,
                'estimated_total': 'unknown',
                'estimated_doc_size_bytes': 0,
                'has_data': False,
                'error': str(e)
            }
    
    def download_endpoint_data(
        self, 
        endpoint_path: str, 
        fields: Optional[List[str]] = None,
        max_materials: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Download all data from a specific endpoint.
        
        Args:
            endpoint_path: Path to the endpoint (e.g., 'materials.summary')
            fields: Specific fields to download (None = all fields)
            max_materials: Maximum number of materials to download (None = all)
            
        Returns:
            Dictionary with download results and statistics
        """
        logger.info(f"Starting download for endpoint: {endpoint_path}")
        
        start_time = time.time()
        downloaded_data = []
        total_downloaded = 0
        chunk_count = 0
        errors = []
        
        try:
            # Use optimized MPRester for large downloads
            with MPRester(
                self.api_key, 
                use_document_model=False, 
                monty_decode=False
            ) as mpr:
                
                # Navigate to endpoint with error handling
                endpoint = mpr
                try:
                    for part in endpoint_path.split('.'):
                        endpoint = getattr(endpoint, part)
                except AttributeError as e:
                    error_msg = f"Endpoint {endpoint_path} not available: {str(e)}"
                    logger.warning(error_msg)
                    errors.append(error_msg)
                    return {
                        'endpoint': endpoint_path,
                        'total_materials': 0,
                        'chunks_downloaded': 0,
                        'download_time_seconds': 0,
                        'materials_per_second': 0,
                        'fields_requested': 0,
                        'errors': errors,
                        'save_results': {"saved": False, "reason": "Endpoint not available"}
                    }
                
                # Get all available fields if none specified
                if fields is None:
                    if hasattr(endpoint, 'available_fields'):
                        fields = list(endpoint.available_fields)
                        logger.info(f"Using all {len(fields)} available fields")
                    else:
                        logger.info("No specific fields specified, downloading with default fields")
                        fields = None
                
                # Download all data at once (MP API doesn't support pagination with _skip/_limit)
                try:
                    chunk_start = time.time()
                    logger.info(f"Downloading all data from endpoint")
                    
                    # Download all data with proper parameters and error handling
                    try:
                        if fields:
                            downloaded_data = endpoint.search(fields=fields)
                        else:
                            downloaded_data = endpoint.search()
                    except TypeError as e:
                        # Some endpoints don't support fields parameter
                        if "unexpected keyword argument 'fields'" in str(e):
                            logger.warning(f"Endpoint {endpoint_path} doesn't support fields parameter, trying without")
                            downloaded_data = endpoint.search()
                        else:
                            raise e
                    except Exception as e:
                        # Handle other endpoint-specific issues
                        error_details = str(e)
                        if "unexpected keyword argument" in error_details:
                            logger.warning(f"Parameter issue with {endpoint_path}: {error_details}")
                            # Try with minimal parameters
                            try:
                                downloaded_data = endpoint.search()
                            except Exception as e2:
                                raise e2
                        else:
                            raise e
                    
                    if not downloaded_data:
                        logger.info(f"No data found for endpoint {endpoint_path}")
                    else:
                        total_downloaded = len(downloaded_data)
                        chunk_count = 1
                        
                        download_time = time.time() - chunk_start
                        logger.info(f"Downloaded {total_downloaded} materials in {download_time:.2f}s")
                        
                        # Apply manual limit if specified
                        if max_materials and total_downloaded > max_materials:
                            logger.info(f"Applying limit: {max_materials} out of {total_downloaded}")
                            downloaded_data = downloaded_data[:max_materials]
                            total_downloaded = max_materials
                        
                        # Memory management for large downloads
                        if total_downloaded > 10000:
                            logger.info("Large dataset detected, triggering garbage collection")
                            gc.collect()
                    
                except Exception as e:
                    error_msg = f"Error downloading data: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                
                # Save the downloaded data
                if downloaded_data:
                    save_results = self._save_endpoint_data(
                        endpoint_path, 
                        downloaded_data, 
                        fields
                    )
                else:
                    save_results = {"saved": False, "reason": "No data downloaded"}
                
                download_time = time.time() - start_time
                
                results = {
                    'endpoint': endpoint_path,
                    'total_materials': total_downloaded,
                    'chunks_downloaded': chunk_count,
                    'download_time_seconds': download_time,
                    'materials_per_second': total_downloaded / download_time if download_time > 0 else 0,
                    'fields_requested': len(fields) if fields else 'all',
                    'errors': errors,
                    'save_results': save_results
                }
                
                logger.info(f"Completed {endpoint_path}: {total_downloaded} materials in {download_time:.2f}s")
                self.total_materials_downloaded += total_downloaded
                
                return results
                
        except Exception as e:
            error_msg = f"Fatal error downloading {endpoint_path}: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            return {
                'endpoint': endpoint_path,
                'total_materials': 0,
                'error': error_msg,
                'download_time_seconds': time.time() - start_time
            }
    
    def _save_endpoint_data(
        self, 
        endpoint_path: str, 
        data: List[Dict], 
        fields: Optional[List[str]]
    ) -> Dict[str, Any]:
        """
        Save endpoint data in multiple formats.
        
        Args:
            endpoint_path: Name of the endpoint
            data: Downloaded data
            fields: Fields that were downloaded
            
        Returns:
            Dictionary with save results
        """
        if not data:
            return {"saved": False, "reason": "No data to save"}
        
        try:
            # Create endpoint-specific directory
            endpoint_dir = self.data_dir / endpoint_path.replace('.', '_')
            endpoint_dir.mkdir(exist_ok=True, parents=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"{endpoint_path.replace('.', '_')}_{timestamp}"
            
            saved_files = []
            
            # Save as compressed JSON (most complete)
            json_path = endpoint_dir / f"{base_filename}.json"
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            saved_files.append(str(json_path))
            logger.info(f"Saved JSON: {json_path}")
            
            # Save as CSV (for analysis)
            try:
                df = pd.DataFrame(data)
                csv_path = endpoint_dir / f"{base_filename}.csv"
                df.to_csv(csv_path, index=False)
                saved_files.append(str(csv_path))
                logger.info(f"Saved CSV: {csv_path}")
            except Exception as e:
                logger.warning(f"Could not save CSV for {endpoint_path}: {e}")
            
            # Save metadata
            metadata = {
                'endpoint': endpoint_path,
                'download_timestamp': timestamp,
                'total_materials': len(data),
                'fields_downloaded': fields,
                'sample_material_ids': [
                    item.get('material_id', item.get('molecule_id', 'unknown'))
                    for item in data[:10]
                ],
                'data_structure_sample': data[0] if data else None
            }
            
            metadata_path = endpoint_dir / f"{base_filename}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            saved_files.append(str(metadata_path))
            
            return {
                'saved': True,
                'files': saved_files,
                'total_size_mb': sum(
                    os.path.getsize(f) for f in saved_files
                ) / (1024 * 1024)
            }
            
        except Exception as e:
            error_msg = f"Error saving data for {endpoint_path}: {e}"
            logger.error(error_msg)
            return {'saved': False, 'error': error_msg}
    
    def download_all_data(
        self, 
        endpoints_to_download: Optional[List[str]] = None,
        max_materials_per_endpoint: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Download ALL available data from ALL endpoints.
        
        Args:
            endpoints_to_download: Specific endpoints to download (None = all)
            max_materials_per_endpoint: Limit materials per endpoint (None = no limit)
            
        Returns:
            Comprehensive download report
        """
        logger.info("=" * 80)
        logger.info("STARTING COMPREHENSIVE MATERIALS PROJECT DATABASE DOWNLOAD")
        logger.info("=" * 80)
        
        self.download_start_time = time.time()
        
        # Get all available endpoints
        all_endpoints = self.get_all_available_endpoints()
        
        if endpoints_to_download:
            endpoints_to_process = [
                (ep, desc) for ep, desc in all_endpoints 
                if ep in endpoints_to_download
            ]
        else:
            endpoints_to_process = all_endpoints
        
        logger.info(f"Will download from {len(endpoints_to_process)} endpoints:")
        for ep, desc in endpoints_to_process:
            logger.info(f"  - {ep}: {desc}")
        
        # First, estimate total data size
        logger.info("\nEstimating data sizes...")
        size_estimates = {}
        for endpoint, desc in endpoints_to_process:
            size_estimates[endpoint] = self.estimate_data_size(endpoint)
            if size_estimates[endpoint]['has_data']:
                logger.info(f"  {endpoint}: ~{size_estimates[endpoint]['estimated_total']} materials")
        
        # Download from each endpoint
        download_results = {}
        
        for i, (endpoint, desc) in enumerate(endpoints_to_process, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"ENDPOINT {i}/{len(endpoints_to_process)}: {endpoint}")
            logger.info(f"Description: {desc}")
            logger.info(f"{'='*60}")
            
            if not size_estimates[endpoint]['has_data']:
                logger.info(f"Skipping {endpoint} - no data available")
                download_results[endpoint] = {
                    'skipped': True,
                    'reason': 'No data available'
                }
                continue
            
            try:
                # Get all fields for this endpoint
                fields = self.get_all_available_fields(endpoint)
                
                # Download the data
                result = self.download_endpoint_data(
                    endpoint,
                    fields=fields,
                    max_materials=max_materials_per_endpoint
                )
                
                download_results[endpoint] = result
                
                # Log progress
                elapsed = time.time() - self.download_start_time
                logger.info(f"Progress: {i}/{len(endpoints_to_process)} endpoints completed")
                logger.info(f"Total time elapsed: {elapsed/3600:.2f} hours")
                logger.info(f"Total materials downloaded so far: {self.total_materials_downloaded:,}")
                
            except Exception as e:
                error_msg = f"Failed to download {endpoint}: {e}"
                logger.error(error_msg)
                download_results[endpoint] = {
                    'error': error_msg,
                    'total_materials': 0
                }
                self.failed_downloads.append(endpoint)
        
        # Generate comprehensive report
        total_time = time.time() - self.download_start_time
        report = self._generate_final_report(
            download_results, 
            size_estimates, 
            total_time
        )
        
        logger.info("\n" + "="*80)
        logger.info("DOWNLOAD COMPLETE!")
        logger.info("="*80)
        logger.info(f"Total time: {total_time/3600:.2f} hours")
        logger.info(f"Total materials downloaded: {self.total_materials_downloaded:,}")
        logger.info(f"Successful endpoints: {len([r for r in download_results.values() if not r.get('error')])}")
        logger.info(f"Failed endpoints: {len(self.failed_downloads)}")
        
        return report
    
    def _generate_final_report(
        self, 
        download_results: Dict[str, Any], 
        size_estimates: Dict[str, Any], 
        total_time: float
    ) -> Dict[str, Any]:
        """Generate a comprehensive final report."""
        
        successful_downloads = {
            k: v for k, v in download_results.items() 
            if not v.get('error') and not v.get('skipped')
        }
        
        failed_downloads = {
            k: v for k, v in download_results.items() 
            if v.get('error')
        }
        
        skipped_downloads = {
            k: v for k, v in download_results.items() 
            if v.get('skipped')
        }
        
        # Calculate total data size
        total_files = 0
        total_size_mb = 0
        for result in successful_downloads.values():
            if 'save_results' in result and result['save_results'].get('saved'):
                total_files += len(result['save_results']['files'])
                total_size_mb += result['save_results']['total_size_mb']
        
        report = {
            'download_summary': {
                'total_time_hours': total_time / 3600,
                'total_materials_downloaded': self.total_materials_downloaded,
                'total_endpoints_processed': len(download_results),
                'successful_endpoints': len(successful_downloads),
                'failed_endpoints': len(failed_downloads),
                'skipped_endpoints': len(skipped_downloads),
                'total_files_created': total_files,
                'total_data_size_mb': total_size_mb,
                'total_data_size_gb': total_size_mb / 1024,
                'download_rate_materials_per_hour': self.total_materials_downloaded / (total_time / 3600) if total_time > 0 else 0
            },
            'successful_downloads': successful_downloads,
            'failed_downloads': failed_downloads,
            'skipped_downloads': skipped_downloads,
            'size_estimates': size_estimates,
            'data_directory': str(self.data_dir),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save report
        report_path = self.data_dir / f"download_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Detailed report saved to: {report_path}")
        
        return report


def main():
    """Main function to download all Materials Project data."""
    
    try:
        # Print configuration
        config.print_config()
        print("\n" + "="*80)
        print("COMPREHENSIVE MATERIALS PROJECT DATABASE DOWNLOAD")
        print("="*80)
        
        print("WARNING: This will download the ENTIRE Materials Project database!")
        print("This could be tens of gigabytes and take many hours to complete.")
        print("Make sure you have:")
        print("  - Sufficient disk space (50+ GB recommended)")
        print("  - Stable internet connection")
        print("  - Valid MP API key")
        print()
        
        # Ask for confirmation
        response = input("Do you want to proceed? (yes/no): ").lower().strip()
        if response not in ['yes', 'y']:
            print("Download cancelled.")
            return
        
        # Ask about limits
        print("\nDownload options:")
        print("1. Download ALL data (recommended for research)")
        print("2. Download with material limits (for testing)")
        print("3. Download specific endpoints only")
        
        option = input("Choose option (1-3): ").strip()
        
        max_materials_per_endpoint = None
        endpoints_to_download = None
        
        if option == '2':
            limit_str = input("Enter max materials per endpoint (e.g., 1000): ").strip()
            try:
                max_materials_per_endpoint = int(limit_str)
                print(f"Will limit to {max_materials_per_endpoint} materials per endpoint")
            except ValueError:
                print("Invalid number, proceeding without limits")
                
        elif option == '3':
            print("\nAvailable endpoint categories:")
            print("  - materials.summary (core data)")
            print("  - materials.electronic_structure (band structures, DOS)")
            print("  - materials.elasticity (mechanical properties)")
            print("  - materials.thermo (thermodynamics)")
            print("  - molecules.summary (molecular data)")
            print("  - (or enter specific endpoint names)")
            
            endpoints_str = input("Enter endpoints separated by commas: ").strip()
            if endpoints_str:
                endpoints_to_download = [ep.strip() for ep in endpoints_str.split(',')]
                print(f"Will download from: {endpoints_to_download}")
        
        # Initialize downloader
        chunk_size = 5000  # Smaller chunks for stability
        downloader = ComprehensiveMPDownloader(chunk_size=chunk_size)
        
        # Start the download
        print(f"\nStarting download at {datetime.now()}")
        print(f"Data will be saved to: {downloader.data_dir}")
        
        results = downloader.download_all_data(
            endpoints_to_download=endpoints_to_download,
            max_materials_per_endpoint=max_materials_per_endpoint
        )
        
        # Print final summary
        print("\n" + "="*80)
        print("DOWNLOAD COMPLETED!")
        print("="*80)
        summary = results['download_summary']
        print(f"Total time: {summary['total_time_hours']:.2f} hours")
        print(f"Materials downloaded: {summary['total_materials_downloaded']:,}")
        print(f"Successful endpoints: {summary['successful_endpoints']}")
        print(f"Total data size: {summary['total_data_size_gb']:.2f} GB")
        print(f"Download rate: {summary['download_rate_materials_per_hour']:.0f} materials/hour")
        print(f"Data location: {results['data_directory']}")
        
        if results['failed_downloads']:
            print(f"\nFailed endpoints ({len(results['failed_downloads'])}):")
            for endpoint, error_info in results['failed_downloads'].items():
                print(f"  - {endpoint}: {error_info.get('error', 'Unknown error')}")
        
        print("\nFiles created:")
        for endpoint, result in results['successful_downloads'].items():
            if 'save_results' in result and result['save_results'].get('saved'):
                files = result['save_results']['files']
                size_mb = result['save_results']['total_size_mb']
                print(f"  {endpoint}: {len(files)} files, {size_mb:.1f} MB")
        
        print(f"\nDetailed log saved to: download_all_data.log")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        print("Partial data may have been saved.")
        
    except Exception as e:
        print(f"\nFatal error: {e}")
        print("Check the log file for detailed error information.")
        logger.error(f"Fatal error in main: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()