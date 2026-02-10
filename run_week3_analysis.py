"""
Week 3 Explainability Analysis Runner

This script executes the complete Week 3 explainability analysis:
1. Checks for spurious background/lighting patterns
2. Generates visual evidence for the explainability report

Usage:
    python run_week3_analysis.py
"""

import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from explainability.spurious_pattern_check import main as run_spurious_check

def main():
    print("="*70)
    print(" WEEK 3 EXPLAINABILITY ANALYSIS")
    print("="*70)
    print("\nThis script will:")
    print("  1. Analyze Grad-CAM heatmaps for spurious patterns")
    print("  2. Generate visual evidence and statistics")
    print("  3. Create output files for the explainability report")
    print("\n" + "="*70 + "\n")
    
    # Run spurious pattern detection
    print("Starting spurious pattern analysis...")
    print("-"*70 + "\n")
    
    try:
        run_spurious_check()
        
        print("\n" + "="*70)
        print(" ANALYSIS COMPLETE!")
        print("="*70)
        print("\n📊 Results Summary:")
        print("  ✓ Visualization outputs: assets/sample_outputs/spurious_check/*.png")
        print("  ✓ Analysis data: assets/sample_outputs/spurious_check/spurious_pattern_analysis.json")
        print("\n📝 Next Step:")
        print("  → Update DOCS/EXPLAINABILITY_REPORT.md with the analysis results")
        print("  → Review the generated visualizations")
        print("  → Fill in the [TO BE UPDATED] sections with actual data\n")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure your trained model exists in models/trained/")
        print("  2. Check that test data is available in data/splits/test/")
        print("  3. Verify Grad-CAM implementation in src/explainability/gradcam.py")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
