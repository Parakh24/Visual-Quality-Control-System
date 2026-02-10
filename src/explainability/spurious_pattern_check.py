"""
Spurious Pattern Detection using Grad-CAM Analysis

This script analyzes whether the model is learning spurious patterns
(background, lighting, irrelevant features) instead of actual defects.

Week 3 Task: Ensure the model is not learning spurious background or lighting patterns.

Author: Vision QC Team - Week 3
"""

import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import config
from explainability.gradcam import generate_gradcam, load_and_preprocess_image
from explainability.visualize import overlay_heatmap


class SpuriousPatternAnalyzer:
    """Analyzes Grad-CAM outputs to detect spurious pattern learning"""
    
    def __init__(self, model_path, last_conv_layer_name="top_conv"):
        """
        Initialize the analyzer
        
        Args:
            model_path: Path to trained model (.h5 or .keras)
            last_conv_layer_name: Name of last conv layer for Grad-CAM
        """
        self.model = tf.keras.models.load_model(model_path)
        self.last_conv_layer = last_conv_layer_name
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "model_path": model_path,
            "images_analyzed": [],
            "summary": {}
        }
        
    def analyze_heatmap_distribution(self, heatmap, threshold=0.5):
        """
        Analyze spatial distribution of heatmap activations
        
        Returns:
            dict: Statistics about heatmap distribution
        """
        # Get high-activation regions
        high_activation_mask = heatmap > threshold
        high_activation_ratio = np.sum(high_activation_mask) / heatmap.size
        
        # Calculate center of mass of activations
        if np.sum(heatmap) > 0:
            y_indices, x_indices = np.where(heatmap > 0)
            weights = heatmap[y_indices, x_indices]
            center_y = np.average(y_indices, weights=weights) / heatmap.shape[0]
            center_x = np.average(x_indices, weights=weights) / heatmap.shape[1]
        else:
            center_y, center_x = 0.5, 0.5
        
        # Check if activations are at edges (potential background focus)
        edge_thickness = int(heatmap.shape[0] * 0.1)  # 10% from edges
        edge_mask = np.zeros_like(heatmap, dtype=bool)
        edge_mask[:edge_thickness, :] = True
        edge_mask[-edge_thickness:, :] = True
        edge_mask[:, :edge_thickness] = True
        edge_mask[:, -edge_thickness:] = True
        
        edge_activation = np.sum(heatmap * edge_mask) / np.sum(heatmap) if np.sum(heatmap) > 0 else 0
        
        # Calculate concentration (how focused vs diffuse)
        max_activation = np.max(heatmap)
        mean_activation = np.mean(heatmap[heatmap > 0]) if np.any(heatmap > 0) else 0
        concentration = max_activation - mean_activation if max_activation > 0 else 0
        
        return {
            "high_activation_ratio": float(high_activation_ratio),
            "center_of_mass": {"x": float(center_x), "y": float(center_y)},
            "edge_activation_ratio": float(edge_activation),
            "concentration": float(concentration),
            "max_activation": float(max_activation),
            "mean_activation": float(mean_activation)
        }
    
    def check_for_spurious_patterns(self, stats):
        """
        Determine if heatmap shows signs of spurious pattern learning
        
        Returns:
            dict: Assessment results
        """
        issues = []
        warnings = []
        
        # Check 1: Too much edge activation (background focus)
        if stats["edge_activation_ratio"] > 0.3:
            issues.append("HIGH_EDGE_ACTIVATION: Model may be focusing on background/borders")
        elif stats["edge_activation_ratio"] > 0.2:
            warnings.append("MODERATE_EDGE_ACTIVATION: Some background attention detected")
        
        # Check 2: Too diffuse activation (no clear focus)
        if stats["concentration"] < 0.2:
            issues.append("LOW_CONCENTRATION: Activations too diffuse, no clear defect focus")
        elif stats["concentration"] < 0.35:
            warnings.append("MODERATE_CONCENTRATION: Could be more focused on defects")
        
        # Check 3: Too much high activation area (global pattern)
        if stats["high_activation_ratio"] > 0.5:
            issues.append("GLOBAL_ACTIVATION: Model activating on too large area (>50%)")
        elif stats["high_activation_ratio"] > 0.35:
            warnings.append("BROAD_ACTIVATION: Activation area somewhat broad (>35%)")
        
        # Overall assessment
        if len(issues) == 0:
            if len(warnings) == 0:
                assessment = "GOOD"
                message = "✅ Model appears to focus on localized features (likely defects)"
            else:
                assessment = "ACCEPTABLE"
                message = "⚠️ Model shows acceptable focus with minor concerns"
        else:
            assessment = "POOR"
            message = "❌ Model may be learning spurious patterns"
        
        return {
            "assessment": assessment,
            "message": message,
            "issues": issues,
            "warnings": warnings
        }
    
    def analyze_image(self, image_path, class_index=None, save_output=True):
        """
        Analyze a single image for spurious patterns
        
        Args:
            image_path: Path to image
            class_index: Target class (None = predicted class)
            save_output: Whether to save visualization
            
        Returns:
            dict: Analysis results
        """
        print(f"\n{'='*60}")
        print(f"Analyzing: {os.path.basename(image_path)}")
        print(f"{'='*60}")
        
        # Load and preprocess image
        img_array = load_and_preprocess_image(image_path, config.IMG_SIZE)
        
        # Get prediction
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        # Generate Grad-CAM
        heatmap = generate_gradcam(
            self.model, 
            img_array, 
            self.last_conv_layer,
            class_index=class_index if class_index is not None else predicted_class
        )
        
        # Analyze heatmap distribution
        stats = self.analyze_heatmap_distribution(heatmap)
        assessment = self.check_for_spurious_patterns(stats)
        
        # Print results
        print(f"Prediction: Class {predicted_class} (confidence: {confidence:.2%})")
        print(f"\nHeatmap Statistics:")
        print(f"  - High activation area: {stats['high_activation_ratio']:.2%}")
        print(f"  - Edge activation: {stats['edge_activation_ratio']:.2%}")
        print(f"  - Concentration: {stats['concentration']:.3f}")
        print(f"  - Center of mass: ({stats['center_of_mass']['x']:.2f}, {stats['center_of_mass']['y']:.2f})")
        print(f"\nAssessment: {assessment['assessment']}")
        print(f"{assessment['message']}")
        
        if assessment['issues']:
            print(f"\n🚨 Issues:")
            for issue in assessment['issues']:
                print(f"   - {issue}")
        
        if assessment['warnings']:
            print(f"\n⚠️ Warnings:")
            for warning in assessment['warnings']:
                print(f"   - {warning}")
        
        # Save visualization if requested
        output_path = None
        if save_output:
            output_dir = os.path.join(config.BASE_DIR, "assets", "sample_outputs", "spurious_check")
            os.makedirs(output_dir, exist_ok=True)
            
            # Load original image for overlay
            original_img = cv2.imread(image_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            original_img = cv2.resize(original_img, config.IMG_SIZE)
            
            # Create overlay
            overlay = overlay_heatmap(original_img, heatmap, alpha=0.4)
            
            # Save
            filename = f"{Path(image_path).stem}_spurious_check.png"
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            print(f"\n💾 Saved visualization: {output_path}")
        
        # Store results
        result = {
            "image_path": image_path,
            "predicted_class": int(predicted_class),
            "confidence": confidence,
            "statistics": stats,
            "assessment": assessment,
            "output_path": output_path
        }
        
        self.results["images_analyzed"].append(result)
        
        return result
    
    def analyze_dataset(self, data_dir, max_samples_per_class=10):
        """
        Analyze multiple images from dataset
        
        Args:
            data_dir: Directory containing images (assumes class subdirectories)
            max_samples_per_class: Maximum samples to analyze per class
        """
        print(f"\n{'#'*60}")
        print(f"# SPURIOUS PATTERN DETECTION ANALYSIS")
        print(f"{'#'*60}\n")
        
        # Find all images
        image_paths = []
        for class_dir in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_dir)
            if os.path.isdir(class_path):
                images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
                images = images[:max_samples_per_class]
                image_paths.extend([os.path.join(class_path, img) for img in images])
        
        # Analyze each image
        for img_path in image_paths:
            self.analyze_image(img_path, save_output=True)
        
        # Generate summary
        self._generate_summary()
        
        # Save results
        self._save_results()
    
    def _generate_summary(self):
        """Generate summary statistics across all analyzed images"""
        if not self.results["images_analyzed"]:
            return
        
        assessments = [img["assessment"]["assessment"] for img in self.results["images_analyzed"]]
        
        good_count = assessments.count("GOOD")
        acceptable_count = assessments.count("ACCEPTABLE")
        poor_count = assessments.count("POOR")
        total = len(assessments)
        
        # Collect all issues
        all_issues = []
        all_warnings = []
        for img in self.results["images_analyzed"]:
            all_issues.extend(img["assessment"]["issues"])
            all_warnings.extend(img["assessment"]["warnings"])
        
        # Count issue types
        from collections import Counter
        issue_counts = Counter(all_issues)
        warning_counts = Counter(all_warnings)
        
        # Overall verdict
        if poor_count / total > 0.3:
            verdict = "CONCERNING"
            message = "❌ Significant evidence of spurious pattern learning detected"
        elif poor_count / total > 0.1 or acceptable_count / total > 0.5:
            verdict = "NEEDS_ATTENTION"
            message = "⚠️ Some spurious patterns detected, model may need improvement"
        else:
            verdict = "TRUSTWORTHY"
            message = "✅ Model appears to be learning meaningful features"
        
        summary = {
            "total_images": total,
            "good": good_count,
            "acceptable": acceptable_count,
            "poor": poor_count,
            "good_percentage": round(good_count / total * 100, 2),
            "acceptable_percentage": round(acceptable_count / total * 100, 2),
            "poor_percentage": round(poor_count / total * 100, 2),
            "verdict": verdict,
            "message": message,
            "common_issues": dict(issue_counts),
            "common_warnings": dict(warning_counts)
        }
        
        self.results["summary"] = summary
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"Total Images Analyzed: {total}")
        print(f"  ✅ Good: {good_count} ({summary['good_percentage']}%)")
        print(f"  ⚠️ Acceptable: {acceptable_count} ({summary['acceptable_percentage']}%)")
        print(f"  ❌ Poor: {poor_count} ({summary['poor_percentage']}%)")
        print(f"\nOverall Verdict: {verdict}")
        print(f"{message}")
        
        if issue_counts:
            print(f"\nMost Common Issues:")
            for issue, count in issue_counts.most_common(3):
                print(f"  - {issue}: {count} occurrences")
        
        if warning_counts:
            print(f"\nMost Common Warnings:")
            for warning, count in warning_counts.most_common(3):
                print(f"  - {warning}: {count} occurrences")
    
    def _save_results(self):
        """Save analysis results to JSON"""
        output_dir = os.path.join(config.BASE_DIR, "assets", "sample_outputs", "spurious_check")
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, "spurious_pattern_analysis.json")
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Full results saved to: {output_path}")


def main():
    """Main execution function"""
    # Configuration
    model_path = os.path.join(config.TRAINED_MODELS_DIR, "vision_spec_qc.keras")
    test_data_dir = config.TEST_DIR
    
    # Check if model exists
    if not os.path.exists(model_path):
        # Try .h5 extension
        model_path = os.path.join(config.TRAINED_MODELS_DIR, "vision_spec_qc.h5")
        if not os.path.exists(model_path):
            print(f"❌ Model not found at {model_path}")
            print("Please ensure your trained model is in the correct location.")
            return
    
    # Check if test data exists
    if not os.path.exists(test_data_dir):
        print(f"❌ Test data directory not found: {test_data_dir}")
        return
    
    # Create analyzer
    analyzer = SpuriousPatternAnalyzer(
        model_path=model_path,
        last_conv_layer_name="top_conv"  # Adjust based on your model architecture
    )
    
    # Analyze test dataset
    analyzer.analyze_dataset(
        data_dir=test_data_dir,
        max_samples_per_class=15  # Analyze 15 images per class
    )
    
    print(f"\n{'#'*60}")
    print("# Analysis complete!")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
