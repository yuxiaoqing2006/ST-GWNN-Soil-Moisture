# ST-GWNN Experimental Framework
Complete ablation and baseline comparison experimental framework to verify the effectiveness of the ST-GWNN model.
## 📁 Directory Structure

## Quick‑Test (sanity check, no real dataset needed)
You can run a lightweight sanity verification without downloading original soil‑moisture data:
```bash
python quick_test.py

## 🎯 Experimental Objectives
1. **Ablation Study**: Verify the effectiveness of each component of ST-GWNN
2. **Baseline Comparison**: Compare with classic deep learning models to demonstrate the superiority of ST-GWNN
3. **Statistical Analysis**: Validate the significance of performance improvements via paired t-test
## 🔬 Experimental Design
### 1. Ablation Study
Remove key components of ST-GWNN and observe performance variations:
| Model Variant | Description | Expected Impact |
|---------|------|---------|
| **Full_Model** | Complete ST-GWNN model | Baseline (R²≈0.80) |
| **No_Wavelet** | Remove graph wavelet transform | Degraded spatial feature extraction capability |
| **No_Temporal** | Remove temporal gated convolution | Degraded time-series modeling capability |
| **No_Attention** | Remove spatial attention | Impaired ability to learn dynamic relationships |
| **No_Residual** | Remove residual connections | Hindered gradient flow, difficult training |
**Key Questions**:
- Which component has the largest impact on performance?
- What is the contribution ranking of each component?
- Is the performance drop after component removal statistically significant?

### 2. Baseline Comparison Experiment
Comparison with classic deep learning models:
| Baseline Model | Characteristics | Expected Performance |
|---------|------|---------|
| **LSTM** | Classical time-series model, ignores spatial relationships | R²≈0.65-0.70 |
| **GCN** | Graph convolutional network with simple temporal aggregation | R²≈0.70-0.75 |
| **Transformer** | Self-attention mechanism for joint spatio-temporal modeling | R²≈0.72-0.77 |
**Key Questions**:
- What is the magnitude of performance improvement of ST-GWNN over baseline models?
- Is the improvement statistically significant?
- What is the parameter efficiency of different models?

## 🚀 Quick Start
### Option 1: Run all experiments with one click (Recommended)
```bash
cd ST_GWNN_SoilMoisture/experiments
python run_all_experiments.py
```
This sequentially executes:
1. Ablation study (approx. 30–60 minutes)
2. Baseline comparison experiments (approx. 30–60 minutes)
3. Result analysis and visualization (approx. 1–2 minutes)



### Option 2: Run step-by-step
```bash
# 1. Run ablation experiments
python run_ablation_experiments.py
# 2. Run baseline comparison experiments
python run_baseline_experiments.py
# 3. Analyze results
python analyze_all_experiments.py
```
## 📊 Output Results
### 1. Numerical Results
**Ablation experiment results**: `../results/ablation/ablation_results.json`
```json
{
  "Full_Model": {
    "params": 380000,
    "test_r2": 0.8040,
    "test_rmse": 0.0234,
    "generalization_gap": 0.0419
  },
  "No_Wavelet": {...},
  ...
}
```
**Baseline comparison results**: `../results/baseline/baseline_results.json`
```json
{
  "LSTM": {
    "params": 250000,
    "test_r2": 0.6800,
    "test_rmse": 0.0298
  },
  ...
}
```
**Comprehensive comparison table**: `../results/all_experiments_comparison.csv`

### 2. Visualization Results
Generated figures are saved under `../results/`:
1. **comprehensive_comparison.png** - Comprehensive performance comparison
   - Test R² comparison
   - Parameter count comparison
   - Generalization gap comparison
   - Training efficiency comparison
2. **ablation_analysis.png** - Ablation experiment analysis
   - Performance impact of each component
   - Component importance ranking
3. **baseline_comparison.png** - Baseline model comparison
   - ST-GWNN vs LSTM/GCN/Transformer
   - Percentage of performance improvement

### 3. Statistical Analysis
**Paired t-test results**: Verify whether the performance improvement of ST-GWNN over baseline models is significant
```
ST-GWNN vs Baseline Models (Paired t-test):
  vs LSTM: t=-15.2341, p=2.34e-45 ***
  vs GCN: t=-12.8765, p=1.23e-38 ***
  vs Transformer: t=-10.4532, p=5.67e-32 ***
Significance level: *** p<0.001, ** p<0.01, * p<0.05, ns not significant
```

### 4. Experiment Summary Report
**experiment_summary.txt** - Automatically generated experiment summary:
- Best model and its performance
- Key findings from ablation experiments
- Key findings from baseline comparison
- Model efficiency analysis
- Generalization capability analysis

## 📈 Expected Results
### Expected Ablation Experiment Results
| Model | Test R² | Relative to Full Model | Component Importance |
|------|--------|-------------|-----------|
| Full_Model | 0.8040 | - | - |
| No_Wavelet | 0.7650 | -0.0390 (-4.9%) | ⭐⭐⭐⭐ |
| No_Temporal | 0.7520 | -0.0520 (-6.5%) | ⭐⭐⭐⭐⭐ |
| No_Attention | 0.7780 | -0.0260 (-3.2%) | ⭐⭐⭐ |
| No_Residual | 0.7850 | -0.0190 (-2.4%) | ⭐⭐ |
**Key Findings**:
- Temporal gated convolution contributes the most (6.5% performance drop upon removal)
- Graph wavelet transform ranks second (4.9% performance drop upon removal)
- All components contribute significantly to model performance

### Expected Baseline Comparison Results
| Model | Test R² | Parameter Count | Improvement over ST-GWNN |
|------|--------|--------|------------|
| LSTM | 0.6800 | 250K | +0.1240 (+18.2%) |
| GCN | 0.7200 | 180K | +0.0840 (+11.7%) |
| Transformer | 0.7500 | 420K | +0.0540 (+7.2%) |
| **ST-GWNN** | **0.8040** | **380K** | - |
**Key Findings**:
- ST-GWNN significantly outperforms all baseline models
- 18.2% improvement over LSTM, demonstrating the importance of spatial modeling
- 11.7% improvement over GCN, demonstrating the importance of temporal modeling
- 7.2% improvement over Transformer, demonstrating the advantages of graph wavelets

## 🔧 Customizing Experiments
### Modify experiment configuration
Edit configurations in each experiment script:
```python
model_config = {
    'num_nodes': 7,
    'input_dim': 7,
    'hidden_dim': 136,      # Adjustable
    'output_dim': 1,
    'seq_len': 30,
    'num_gwnn_layers': 3,   # Adjustable
    'num_temporal_blocks': 3,  # Adjustable
    'dropout': 0.1          # Adjustable
}
# Training configuration
num_epochs = 150           # Adjustable
patience = 40              # Adjustable
batch_size = 32            # Adjustable
```
### Add new ablation variants
Add new model classes in `ablation/models_ablation.py`:
```python
class ST_GWNN_NoYourComponent(nn.Module):
    """Variant with your target component removed"""
    def __init__(self, config):
        super().__init__()
        # Implement your variant
        ...
```
Then register it in `run_ablation_experiments.py`:
```python
experiments = {
    'Full_Model': ST_GWNN_Optimized,
    'No_YourComponent': ST_GWNN_NoYourComponent,
    ...
}
```
### Add new baseline models
Add new models in `baseline/models_baseline.py`, then register them in `run_baseline_experiments.py`.

## 📝 Notes
1. **Data Requirements**:
   - Ensure data preprocessing has been executed: `../data/preprocessed_block_split/`
   - Required files: `X_train.npy`, `y_train.npy`, `X_val.npy`, `y_val.npy`, `X_test.npy`, `y_test.npy`, `config.json`
2. **Computational Resources**:
   - GPU (CUDA) is recommended
   - Full experiment runtime: ~1–2 hours (GPU) or 4–6 hours (CPU)
   - Memory requirement: at least 8GB
3. **Result Reproducibility**:
   - Fix random seeds (add at the start of scripts if needed)
   - Slight fluctuations of ±0.01 under identical configurations are normal
4. **Troubleshooting Failed Experiments**:
   - If a single experiment fails, re-run it individually
   - Verify correctness of data paths
   - Check for sufficient GPU memory

## 🎓 Citation
If you use this experimental framework, please cite:
```bibtex
@article{your_paper,
  title={ST-GWNN: Spatio-Temporal Graph Wavelet Neural Network for Soil Moisture Prediction},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```
## 📧 Contact
For questions, please contact: [your-email@example.com]
## 📄 License
MIT License
