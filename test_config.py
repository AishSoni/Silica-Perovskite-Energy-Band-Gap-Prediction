"""
Test script to verify your Materials Project API configuration.
Run this script to check if your .env file is properly configured.
"""

from config import config
from mp_api.client import MPRester

def test_configuration():
    """Test the Materials Project API configuration."""
    
    print("Testing Materials Project API Configuration")
    print("=" * 50)
    
    # Print configuration (without revealing API key)
    config.print_config()
    print()
    
    # Test API key validation
    try:
        api_key = config.get_api_key()
        print("✓ API key is configured")
        
        # Test API connection
        print("Testing API connection...")
        with MPRester(api_key) as mpr:
            # Try a simple search to test the connection
            docs = mpr.materials.summary.search(
                elements=["Si"],
                energy_above_hull=(None, 0.01),
                fields=["material_id", "formula_pretty"]
            )
            
            if docs:
                print(f"✓ API connection successful")
                print(f"  Test query returned: {docs[0].formula_pretty} ({docs[0].material_id})")
                print(f"  Found {len(docs)} total silicon materials")
            else:
                print("⚠ API connection works but no materials found in test query")
                
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
        print("\nTo fix this:")
        print("1. Open the .env file in your project directory")
        print("2. Set your API key: MP_API_KEY=your_actual_api_key_here")
        print("3. Get your API key from: https://materialsproject.org/api")
        return False
        
    except Exception as e:
        print(f"✗ API connection failed: {e}")
        print("\nPossible issues:")
        print("- Check your internet connection")
        print("- Verify your API key is correct")
        print("- Check if Materials Project API is experiencing issues")
        return False
    
    print("\n✓ All tests passed! Your configuration is working correctly.")
    return True

if __name__ == "__main__":
    test_configuration()