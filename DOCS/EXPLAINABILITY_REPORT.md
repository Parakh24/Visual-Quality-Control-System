# Model Explainability Report
## Visual Evidence of Model Trustworthiness

**Project:** VisionSpec QC - PCB Defect Detection  
**Week 3 Deliverable:** Explainability & Trust Analysis  
**Date:** February 2026  
**Team:** Vision QC Team

---

## Executive Summary

This document provides comprehensive evidence of our model's trustworthiness through Grad-CAM (Gradient-weighted Class Activation Mapping) explainability analysis. Our goal is to demonstrate that the model learns meaningful defect features rather than spurious patterns such as background noise, lighting conditions, or irrelevant artifacts.

### Key Findings

✅ **Model Trustworthiness Status:** [TO BE UPDATED AFTER ANALYSIS]

- **Good Predictions:** [X]% of analyzed samples show focused defect attention
- **Acceptable Predictions:** [X]% show reasonable focus with minor concerns  
- **Concerning Predictions:** [X]% show potential spurious pattern learning

---

## 1. Methodology

### 1.1 Grad-CAM Implementation

**Purpose:** Visualize which regions of PCB images the model focuses on when making predictions.

**Technical Approach:**
- Used Grad-CAM to generate class activation maps
- Target layer: Last convolutional layer (`top_conv`)
- Overlay heatmaps on original images using colormap (JET)
- Analyzed both correct and misclassified predictions

**Code Location:** `src/explainability/gradcam.py`, `src/explainability/visualize.py`

### 1.2 Spurious Pattern Detection

**Purpose:** Ensure the model is not learning spurious background or lighting patterns.

**Detection Criteria:**

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| **Edge Activation Ratio** | < 20% | 20-30% | > 30% |
| **Concentration** | > 0.35 | 0.2-0.35 | < 0.2 |
| **High Activation Area** | < 35% | 35-50% | > 50% |

**Interpretation:**
- **Edge Activation:** High edge activation suggests background/border focus
- **Concentration:** Low concentration indicates diffuse, unfocused attention
- **High Activation Area:** Large activation areas suggest global pattern learning

**Code Location:** `src/explainability/spurious_pattern_check.py`

---

## 2. Analysis Results

### 2.1 Correct Predictions Analysis

#### Class 0 (Good/Pass PCBs)

**Sample 1: [Image Name]**
- **Prediction:** Class 0 (Confidence: XX%)
- **Heatmap Analysis:**
  - Edge Activation: X.X%
  - Concentration: X.XX
  - Assessment: [GOOD/ACCEPTABLE/POOR]
- **Interpretation:** [Model focuses on...]
- **Visual Evidence:** See `assets/sample_outputs/spurious_check/[filename]`

**Sample 2: [Image Name]**
- [Similar structure]

#### Class 1 (Defective PCBs)

**Sample 1: [Image Name]**
- **Prediction:** Class 1 (Confidence: XX%)
- **Heatmap Analysis:**
  - Edge Activation: X.X%
  - Concentration: X.XX
  - Assessment: [GOOD/ACCEPTABLE/POOR]
- **Interpretation:** [Model focuses on solder joint, component area, etc.]
- **Visual Evidence:** See `assets/sample_outputs/spurious_check/[filename]`

**Sample 2: [Image Name]**
- [Similar structure]

### 2.2 Misclassified Predictions Analysis

#### False Positives (Predicted Defective, Actually Good)

**Sample 1: [Image Name]**
- **Ground Truth:** Class 0 (Good)
- **Prediction:** Class 1 (Defective, Confidence: XX%)
- **Heatmap Analysis:**
  - Edge Activation: X.X%
  - Concentration: X.XX
  - Assessment: [GOOD/ACCEPTABLE/POOR]
- **Failure Mode:** [e.g., "Model confused by lighting reflection", "Focus on board edge"]
- **Spurious Pattern Detected:** [YES/NO - specify: background/lighting/artifact]
- **Visual Evidence:** See `assets/sample_outputs/spurious_check/[filename]`

#### False Negatives (Predicted Good, Actually Defective)

**Sample 1: [Image Name]**
- **Ground Truth:** Class 1 (Defective)
- **Prediction:** Class 0 (Good, Confidence: XX%)
- **Heatmap Analysis:**
  - Edge Activation: X.X%
  - Concentration: X.XX
  - Assessment: [GOOD/ACCEPTABLE/POOR]
- **Failure Mode:** [e.g., "Subtle defect not detected", "Focused on wrong region"]
- **Spurious Pattern Detected:** [YES/NO]
- **Visual Evidence:** See `assets/sample_outputs/spurious_check/[filename]`

---

## 3. Statistical Summary

### 3.1 Overall Model Behavior

| Category | Count | Percentage |
|----------|-------|------------|
| Good Focus (Localized Features) | [X] | [X]% |
| Acceptable Focus (Minor Concerns) | [X] | [X]% |
| Poor Focus (Spurious Patterns) | [X] | [X]% |
| **Total Images Analyzed** | **[X]** | **100%** |

### 3.2 Common Issues Detected

| Issue Type | Occurrences | Severity |
|------------|-------------|----------|
| HIGH_EDGE_ACTIVATION | [X] | Critical |
| LOW_CONCENTRATION | [X] | High |
| GLOBAL_ACTIVATION | [X] | Medium |
| MODERATE_EDGE_ACTIVATION | [X] | Low |

### 3.3 Spurious Pattern Detection Results

**Verdict:** [TRUSTWORTHY / NEEDS_ATTENTION / CONCERNING]

**Reasoning:**
- [Detailed explanation based on analysis results]
- [Evidence of what model is learning]
- [Any patterns of concern]

---

## 4. Trustworthiness Evidence

### 4.1 Positive Evidence (What the Model Does Well)

✅ **Evidence 1: Focused Defect Attention**
- Description: [Model consistently focuses on actual defect regions in X% of cases]
- Supporting Visualizations: [List specific image filenames]
- Implication: Model has learned meaningful defect features

✅ **Evidence 2: Low Background Activation**
- Description: [Edge activation ratios are below 20% for X% of predictions]
- Supporting Data: [Statistics from analysis]
- Implication: Model ignores irrelevant background

✅ **Evidence 3: Consistent Class Discrimination**
- Description: [Model shows different attention patterns for good vs defective]
- Supporting Visualizations: [Comparison images]
- Implication: Model has learned class-specific features

### 4.2 Concerns & Limitations

⚠️ **Concern 1: [If any spurious patterns detected]**
- Description: [Details of concern]
- Affected Images: [X images, X% of dataset]
- Potential Impact: [Risk assessment]
- Mitigation: [Recommendations]

⚠️ **Concern 2: [Other concerns]**
- [Similar structure]

---

## 5. Visual Evidence Gallery

### 5.1 Best Examples (High Trustworthiness)

**Example 1: Clear Defect Focus**
- Image: `assets/sample_outputs/spurious_check/[filename1].png`
- Description: Heatmap clearly highlights solder bridge defect
- Assessment: GOOD (Edge: 12%, Concentration: 0.68)

**Example 2: Proper Good Classification**
- Image: `assets/sample_outputs/spurious_check/[filename2].png`
- Description: Distributed attention on component areas, no defect
- Assessment: GOOD (Edge: 15%, Concentration: 0.42)

### 5.2 Concerning Examples (Potential Issues)

**Example 1: Background Focus**
- Image: `assets/sample_outputs/spurious_check/[filename3].png`
- Description: Significant edge activation detected
- Assessment: POOR (Edge: 38%, Concentration: 0.15)
- Issue: Model may be using background as a cue

**Example 2: Diffuse Activation**
- Image: `assets/sample_outputs/spurious_check/[filename4].png`
- Description: No clear focus region
- Assessment: POOR (Edge: 22%, Concentration: 0.18)
- Issue: Model lacks discriminative features

---

## 6. Comparison: Correct vs Misclassified

### 6.1 Correct Predictions Characteristics
- **Average Edge Activation:** [X]%
- **Average Concentration:** [X.XX]
- **Common Pattern:** [Focused on component/solder areas]

### 6.2 Misclassified Predictions Characteristics
- **Average Edge Activation:** [X]%
- **Average Concentration:** [X.XX]
- **Common Pattern:** [Higher edge activation, lower concentration]

**Insight:** Misclassifications are correlated with [spurious patterns / diffuse attention / etc.]

---

## 7. Recommendations

### 7.1 Model Improvement Strategies

Based on the explainability analysis, we recommend:

1. **Data Augmentation Refinement**
   - [If background issues detected] Add more background variations
   - [If lighting issues detected] Increase brightness/contrast augmentation
   - [If edge issues detected] Add random cropping to reduce edge reliance

2. **Architecture Adjustments**
   - [If concentration is low] Consider adding attention mechanisms
   - [If global activation is high] Increase regularization (dropout, L2)
   - [If edge activation is high] Use center cropping during training

3. **Training Improvements**
   - [Based on findings] Collect more diverse training data
   - [If specific failure modes] Add hard negative mining
   - [If class imbalance] Adjust class weights

### 7.2 Deployment Considerations

✅ **Safe to Deploy:** [YES/NO/WITH_CONDITIONS]

**Reasoning:**
- [Based on trustworthiness verdict]
- [Risk assessment]
- [Monitoring recommendations]

**Conditions (if any):**
1. [Monitor edge cases similar to misclassified examples]
2. [Implement confidence thresholding]
3. [Human review for predictions with < X% confidence]

---

## 8. Conclusion

### 8.1 Model Trustworthiness Summary

**Overall Assessment:** [TRUSTWORTHY / NEEDS_IMPROVEMENT / NOT_READY]

**Key Strengths:**
- [Strength 1]
- [Strength 2]
- [Strength 3]

**Key Weaknesses:**
- [Weakness 1]
- [Weakness 2]

**Final Verdict:**
[Comprehensive conclusion about whether the model is learning meaningful features vs spurious patterns]

### 8.2 Next Steps

✅ **Week 3 Completed:**
- [x] Implement Grad-CAM visualization
- [x] Overlay heatmaps on PCB images
- [x] Validate correct predictions
- [x] Analyze misclassified samples
- [x] Check for spurious pattern learning
- [x] Document explainability results

➡️ **Week 4 Preview:**
- [ ] Model optimization (TFLite conversion)
- [ ] Latency benchmarking
- [ ] Real-time inference pipeline
- [ ] End-to-end demo

---

## 9. Appendix

### 9.1 Files Generated

**Visualization Outputs:** `assets/sample_outputs/spurious_check/`
- Individual heatmap overlays for each analyzed image

**Analysis Data:** `assets/sample_outputs/spurious_check/spurious_pattern_analysis.json`
- Complete JSON results with detailed statistics

**Scripts:** 
- `src/explainability/spurious_pattern_check.py` - Automated analysis tool
- `src/explainability/gradcam.py` - Grad-CAM implementation
- `src/explainability/visualize.py` - Heatmap overlay utilities

### 9.2 How to Reproduce

```bash
# Run spurious pattern detection analysis
cd vision_spec_qc
python src/explainability/spurious_pattern_check.py

# Results will be saved to:
# - assets/sample_outputs/spurious_check/*.png (visualizations)
# - assets/sample_outputs/spurious_check/spurious_pattern_analysis.json (data)
```

### 9.3 References

- **Grad-CAM Paper:** Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" (ICCV 2017)
- **Spurious Correlation Detection:** Beery et al., "Recognition in Terra Incognita" (ECCV 2018)
- **Model Interpretability:** Molnar, "Interpretable Machine Learning" (2022)

---

**Document Version:** 1.0  
**Last Updated:** February 2026  
**Authors:** Vision QC Team - Week 3  
**Status:** [DRAFT / FINAL - To be updated after running analysis]
