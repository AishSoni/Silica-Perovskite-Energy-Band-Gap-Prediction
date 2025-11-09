#!/usr/bin/env python3
"""
Example script demonstrating how to download ALL available attributes 
for Perovskite materials using the enhanced Materials Project downloader.
"""

from download_data import EnhancedMaterialsProjectDownloader
import pandas as pd

def main():
    """Demonstrate different ways to download all attributes."""
    
    print("="*80)
    print("MATERIALS PROJECT - ALL ATTRIBUTES DOWNLOAD EXAMPLE")
    print("="*80)
    print()
    
    try:
        # Initialize the downloader
        downloader = EnhancedMaterialsProjectDownloader()
        
        print("OPTION 1: Download ALL attributes for a small sample of Perovskites")
        print("="*60)
        
        # Method 1: Use the convenience function with a limit
        print("Downloading 10 Perovskite materials with ALL available attributes...")
        df_sample = downloader.download_all_perovskite_attributes(limit=10)
        
        if not df_sample.empty:
            print(f"✓ Sample download complete!")
            print(f"✓ Downloaded {len(df_sample)} materials")
            print(f"✓ Total attributes (columns): {len(df_sample.columns)}")
            print(f"✓ Sample attributes: {list(df_sample.columns[:10])}...")
            print()
        
        print("OPTION 2: Download ALL Perovskites with standard fields vs all attributes")
        print("="*60)
        
        # Method 2: Compare standard vs all attributes
        print("Downloading with standard fields...")
        perovskite_standard = downloader.query_all_perovskite_materials(all_attributes=False)
        
        if perovskite_standard:
            df_standard = downloader.save_data(
                perovskite_standard[:5],  # Just 5 for comparison
                "comparison_standard",
                all_attributes=False
            )
            
            print("Downloading same materials with ALL attributes...")
            df_all_attrs = downloader.save_data(
                perovskite_standard[:5],  # Same materials
                "comparison_all_attributes", 
                all_attributes=True
            )
            
            print(f"\nCOMPARISON RESULTS:")
            print(f"Standard fields:   {len(df_standard.columns)} columns")
            print(f"All attributes:    {len(df_all_attrs.columns)} columns")
            print(f"Additional data:   {len(df_all_attrs.columns) - len(df_standard.columns)} extra attributes")
            print()
        
        print("OPTION 3: Search for specific materials with all attributes")
        print("="*60)
        
        # Method 3: Search for specific composition with all attributes
        print("Searching for BaTiO3 and related materials with all attributes...")
        
        batio3_docs = downloader.search_materials(
            formula="BaTiO3",
            # No fields parameter = gets all available fields
            limit=3
        )
        
        if batio3_docs:
            df_batio3 = downloader.save_data(
                batio3_docs,
                "batio3_all_attributes",
                all_attributes=True
            )
            
            print(f"✓ Found {len(batio3_docs)} BaTiO3-related materials")
            print(f"✓ All attributes saved with {len(df_batio3.columns)} total columns")
            
            # Show some interesting attributes that are only available with all_attributes=True
            interesting_attrs = []
            for col in df_batio3.columns:
                if any(keyword in col.lower() for keyword in ['elastic', 'magnetic', 'dielectric', 'piezo', 'thermal']):
                    interesting_attrs.append(col)
            
            if interesting_attrs:
                print(f"✓ Found specialized attributes: {interesting_attrs[:5]}...")
            print()
        
        print("="*80)
        print("SUMMARY OF ALL ATTRIBUTES DOWNLOAD")
        print("="*80)
        print("✓ All attributes mode downloads EVERY available field from the MP database")
        print("✓ This includes:")
        print("  - Basic properties (formula, structure, energy)")
        print("  - Electronic properties (band gap, DOS, band structure)")
        print("  - Mechanical properties (elastic constants, bulk modulus)")
        print("  - Magnetic properties (magnetic moments, magnetic ordering)")
        print("  - Thermal properties (thermal expansion, heat capacity)")
        print("  - And many more specialized fields")
        print()
        print("✓ Use cases:")
        print("  - Comprehensive materials analysis")
        print("  - Machine learning with maximum feature sets")
        print("  - Research requiring all available experimental/calculated data")
        print()
        print("✓ Consider using limits for initial testing due to:")
        print("  - Larger file sizes")
        print("  - Longer download times")
        print("  - More complex data structures")
        
        print("\n" + "="*80)
        print("FILES CREATED:")
        for file_path in downloader.data_dir.glob("*"):
            if file_path.is_file():
                print(f"  - {file_path.name}")
        print("="*80)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()