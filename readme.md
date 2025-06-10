# Quantum Machine Learning for Bias Mitigation in Criminal Justice: A COMPAS Dataset Analysis

## 🎯 Project Overview

This repository explores the application of **Quantum Machine Learning (QML)** methods to address algorithmic bias in criminal justice prediction systems, specifically using the COMPAS (Correctional Offender Management Profiling for Alternative Sanctions) dataset. The project investigates whether quantum-enhanced models can outperform classical approaches in terms of fairness and equitable prediction.

## 📄 Research Context

> *"The increasing deployment of machine learning systems in high-stakes decision-making processes—particularly within criminal justice—has raised serious concerns about fairness, transparency, and algorithmic bias. The COMPAS dataset, widely used for predicting recidivism risk, has become a paradigmatic case for assessing such concerns. In this work, we investigate the application of Quantum Machine Learning (QML) methods to bias mitigation, aiming to evaluate whether quantum-enhanced models can outperform classical approaches in terms of fairness and equitable prediction.*
> 
> *We implement a suite of QML architectures, including variational quantum classifiers and quantum kernel methods, and compare their performance with standard machine learning baselines on the COMPAS dataset. Evaluation is conducted using a set of well-established fairness metrics—such as statistical parity difference, equal opportunity, and disparate impact—alongside traditional performance indicators like accuracy and F1-score.*
> 
> *Our experimental results indicate that quantum models exhibit competitive performance while, in specific configurations, yielding notable improvements in fairness. In particular, we observe that certain quantum circuits implicitly regularize over-sensitive decision boundaries, leading to more balanced outcomes across sensitive demographic groups.*
> 
> *This study represents one of the first empirical explorations of quantum bias mitigation in socially critical datasets. We conclude by discussing the implications of these findings for the design of equitable AI systems and the potential of QML as a tool for promoting algorithmic justice."*

## 🏗️ Current Implementation

The current codebase provides a **classical baseline** implementation using PyTorch neural networks, which serves as the foundation for comparison with upcoming quantum models.

### Key Features

- **Dataset Processing**: Automated loading and preprocessing of COMPAS dataset from AIF360
- **Bias Analysis**: Comprehensive analysis of class imbalance and demographic fairness
- **Neural Network Models**: Various architectures 
- **Advanced Training**: Early stopping with validation monitoring
- **Fairness Evaluation**: Statistical Parity Difference, Equal Opportunity Difference
- **Composite Metrics**: Harmonic mean of F1-score and fairness components

## 📊 Evaluation Metrics

### Performance Metrics
- **Accuracy**: Overall prediction correctness
- **Precision**: True positive rate
- **Recall**: Sensitivity to positive cases
- **F1-Score**: Harmonic mean of precision and recall

### Fairness Metrics
- **Statistical Parity Difference (SPD)**: Difference in positive prediction rates between groups
- **Equal Opportunity Difference (EOD)**: Difference in true positive rates between groups
- **Composite Metric**: Combined performance and fairness measure

## 🛠️ Setup and Installation

### Prerequisites
- Python 3.11+
- CUDA 12.4 (for GPU acceleration)

### Environment Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd QML_Compas
```

2. **Create virtual environment**:
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
# or
source .venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**:
```bash
# Install PyTorch with CUDA support first
pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies
pip install -r requirements.txt
```

4. **Download COMPAS dataset**:
The script will automatically prompt you to download the required dataset file if missing:
```bash
# Create directory
mkdir -p .venv/Lib/site-packages/aif360/data/raw/compas

# Download dataset
curl -o .venv/Lib/site-packages/aif360/data/raw/compas/compas-scores-two-years.csv https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv
```

## 🚀 Usage

### Running the Analysis

```bash
python Alessandro/compas_analysis_v2.py
```

### Output
The script will:
1. **Load and preprocess** the COMPAS dataset
2. **Analyze class distribution** and demographic fairness
3. **Train neural network models** with early stopping
4. **Evaluate performance** using multiple metrics
5. **Generate visualizations** of class distribution and fairness
6. **Save preprocessed dataset** to `datasets/compas_from_aif360.csv`

### Sample Output
```
=== Model Performance ===
Accuracy: 0.6842
Precision: 0.6654
Recall: 0.7123
F1 Score: 0.6881

=== Fairness Metrics ===
Statistical Parity Difference: -0.1234
Equal Opportunity Difference: 0.0876
Composite Metric (F1 & Fairness): 0.5234
```

## 📁 Project Structure

```
QML_Compas/
├── Alessandro/
│   └── compas_analysis_v2.py      # Main analysis script
├── datasets/
│   └── compas_from_aif360.csv     # Preprocessed dataset (generated)
├── .venv/                         # Virtual environment
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🔮 Future Development: Quantum Implementation

The next phase of this project will include:

- **Variational Quantum Classifiers (VQC)**: Implementing parameterized quantum circuits
- **Quantum Kernel Methods**: Using quantum feature maps for enhanced pattern recognition
- **Quantum Bias Mitigation**: Exploring quantum regularization techniques
- **Comparative Analysis**: Benchmarking quantum vs. classical approaches
- **Fairness-Aware Quantum Training**: Developing quantum-specific bias mitigation strategies

## 🎛️ Model Architecture

### ComplexNN_1 (Current)
```python
- Input Layer: n_features → 16 neurons
- Hidden Layers: 5 layers with AlphaReLU activation
- Dropout: 0.3 probability between layers
- Output Layer: 16 → 1 (sigmoid activation)
- Loss: Binary Cross-Entropy
- Optimizer: Adam (lr=0.01)
```

### Upcoming Quantum Models
- **Quantum Variational Classifier**: Parameterized quantum circuits with classical post-processing
- **Quantum Kernel SVM**: Quantum feature space mapping with classical SVM
- **Hybrid Quantum-Classical**: Combination of quantum feature extraction and classical neural networks

## 📊 Key Dependencies

```
torch==2.6.04                # PyTorch with CUDA
aif360==0.6.1                # AI Fairness 360 toolkit
scikit-learn==1.6.1          # Machine learning utilities
pandas==2.3.0                # Data manipulation
numpy==2.1.3                 # Numerical computing
matplotlib==3.10.3           # Plotting
seaborn==0.13.2              # Statistical visualization
tensorflow==2.19.0           # Deep learning framework
```

## 🔍 Research Questions

1. **Can quantum models achieve better fairness-performance trade-offs than classical approaches?**
2. **Do quantum circuits provide implicit regularization that reduces discriminatory patterns?**
3. **What quantum architectures are most effective for bias mitigation?**
4. **How do quantum noise effects impact fairness in real-world deployments?**

## 🤝 Contributing

This research is part of an ongoing investigation into quantum bias mitigation. Contributions are welcome, particularly in:

- Quantum circuit design for fairness
- Novel fairness metrics for quantum models
- Theoretical analysis of quantum bias mitigation
- Experimental validation on additional datasets

## 📈 Expected Outcomes

Based on preliminary research, we anticipate:
- **Competitive Performance**: Quantum models matching classical accuracy
- **Enhanced Fairness**: Improved demographic parity through quantum regularization
- **Novel Insights**: Understanding of quantum-specific bias mitigation mechanisms
- **Methodological Framework**: Reusable approach for quantum fairness evaluation

## ⚖️ Ethical Considerations

This research addresses critical issues in algorithmic justice:
- **Bias Mitigation**: Reducing discriminatory outcomes in criminal justice
- **Transparency**: Understanding quantum decision-making processes
- **Accountability**: Ensuring responsible deployment of quantum AI systems
- **Social Impact**: Contributing to equitable technological advancement

## 📜 License

This project is developed for research purposes in algorithmic fairness and quantum machine learning. Please cite appropriately if using this code for academic or research purposes.

---

*This project represents ongoing research in quantum machine learning applications to social justice. Results are preliminary and subject to peer review.*