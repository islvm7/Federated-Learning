# Federated Learning for Arabic Handwriting Recognition

A comprehensive study comparing **Centralized Learning** and **Federated Learning** approaches for Arabic handwriting character recognition using two popular datasets.

## 📋 Project Overview

This project investigates the effectiveness of federated learning in Arabic handwriting recognition tasks. It implements and compares centralized and federated training strategies on two datasets:

- **AHCD**: Arabic Handwritten Characters Dataset
- **OIHACDB**: Offline Isolated Handwritten Arabic Characters Database

The project evaluates both **IID (Independent and Identically Distributed)** and **Non-IID** data distributions to understand how federated learning performs under realistic data heterogeneity conditions.

## 🎯 Key Features

- **Dual Learning Paradigms**: Centralized vs. Federated Learning implementations
- **Multiple Datasets**: Support for AHCD and OIHACDB datasets
- **Data Distribution Scenarios**: Both IID and Non-IID (Dirichlet-based) client data distributions
- **CNN Architecture**: Consistent deep learning model across all experiments
- **Comprehensive Comparisons**: Performance metrics and convergence analysis
- **Visualization Tools**: Training history plots and comparison charts

## 📊 Datasets

### AHCD (Arabic Handwritten Characters Dataset)
- **Training samples**: 13,440 images
- **Test samples**: 3,360 images
- **Classes**: 28 Arabic character classes
- **Image size**: 32×32 pixels (grayscale)

### OIHACDB (Offline Isolated Handwritten Arabic Characters Database)
- Similar structure and format to AHCD
- Alternative dataset for validation and comparison

## 🏗️ Project Structure

```
FL_Arabic_Handwriting/
├── src/
│   ├── preprocessing/          # Data preprocessing scripts
│   │   ├── preprocess_ahcd.py
│   │   └── preprocess_oihacdb.py
│   ├── centralized/            # Centralized learning implementations
│   │   ├── centralized_training_ahcd.py
│   │   └── centralized_training_oihacdb.py
│   └── federated/              # Federated learning implementations
│       ├── split_clients_ahcd.py           # Data splitting for federated setup
│       ├── split_clients_oihacdb.py
│       ├── federated_training_ahcd.py      # Federated learning training
│       ├── federated_training_oihacdb.py
│       ├── compare_iid_non_iid_ahcd.py     # IID vs Non-IID comparison
│       ├── compare_iid_non_iid_oihacdb.py
│       ├── final_comparison_ahcd.py        # Centralized vs Federated
│       ├── final_comparison_oihacdb.py
│       └── FINAL_comparaison.py            # Overall results summary
├── results/                    # Training results and plots
├── reports/                    # Documentation and analysis report
└── .gitignore
```

## 🔧 Prerequisites

### Requirements
- Python 3.8+
- TensorFlow 2.x
- NumPy
- scikit-learn
- Pillow (PIL)
- Matplotlib
- pandas

Install dependencies:
```bash
pip install tensorflow numpy scikit-learn pillow matplotlib pandas
```

## 🚀 Usage

### 1. Data Preprocessing

Prepare and normalize the raw datasets:

```bash
cd src/preprocessing

# Preprocess AHCD
python preprocess_ahcd.py

# Preprocess OIHACDB
python preprocess_oihacdb.py
```

This creates normalized `.npz` files in `data/processed/`.

### 2. Centralized Learning

Train using traditional centralized approach:

```bash
cd src/centralized

# Train on AHCD
python centralized_training_ahcd.py

# Train on OIHACDB
python centralized_training_oihacdb.py
```

### 3. Federated Learning Setup

Split data among simulated clients:

```bash
cd src/federated

# Split AHCD data across 10 clients
python split_clients_ahcd.py

# Split OIHACDB data across 10 clients
python split_clients_oihacdb.py
```

This creates IID and Non-IID distributions of client data.

### 4. Federated Training

Run federated learning algorithms:

```bash
cd src/federated

# Federated learning on AHCD
python federated_training_ahcd.py

# Federated learning on OIHACDB
python federated_training_oihacdb.py
```

### 5. Comparisons and Analysis

Generate comprehensive comparisons:

```bash
cd src/federated

# Compare IID vs Non-IID distributions
python compare_iid_non_iid_ahcd.py
python compare_iid_non_iid_oihacdb.py

# Compare Centralized vs Federated approaches
python final_comparison_ahcd.py
python final_comparison_oihacdb.py

# Generate final summary report
python FINAL_comparaison.py
```

## 🧠 Model Architecture

### CNN Architecture

The project uses a consistent Convolutional Neural Network:

```
Input (32×32×1)
    ↓
Conv2D (32 filters, 3×3)
    ↓
MaxPooling2D (2×2)
    ↓
Conv2D (64 filters, 3×3)
    ↓
MaxPooling2D (2×2)
    ↓
Flatten
    ↓
Dense (128, ReLU)
    ↓
Dropout (0.5)
    ↓
Dense (28, Softmax) → Output
```

**Training Hyperparameters:**
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Sparse Categorical Crossentropy
- **Batch Size**: 32-64
- **Epochs**: 10-20
- **Metrics**: Accuracy

## 📈 Federated Learning Configuration

### IID Distribution
- Equal data samples distributed uniformly across clients
- Samples per client: 1,344 (for AHCD with 10 clients)
- Represents ideal scenario with homogeneous data

### Non-IID Distribution
- Dirichlet distribution (α = 0.1) for realistic heterogeneity
- Clients have different label distributions
- Represents real-world federated scenarios
- More challenging for convergence

### FL Parameters
- **Number of Rounds**: 20
- **Local Epochs**: 5
- **Number of Clients**: 10
- **Communication**: Client models aggregated via FedAvg algorithm

## 📊 Expected Results

The project generates:
- **Training metrics**: Accuracy and loss curves for each configuration
- **Convergence plots**: Federated rounds vs. accuracy
- **Comparison charts**: Centralized vs. Federated performance
- **Distribution analysis**: IID vs. Non-IID impact on learning

### Output Files
- `results/`: PNG plots and training histories (`.npz`)
- `reports/`: PDF report with findings and analysis

## 🔬 Key Comparisons

1. **Centralized vs. Federated**
   - Direct performance comparison
   - Communication cost analysis
   - Privacy implications

2. **IID vs. Non-IID in Federated Learning**
   - Impact of data heterogeneity
   - Convergence speed and stability
   - Final model accuracy

3. **Dataset Comparison**
   - AHCD vs. OIHACDB performance
   - Cross-dataset generalization

## 📝 Notes

- All intermediate and processed data is stored locally
- Results and plots are saved in `results/` for analysis

## 📄 Related Documentation

- Detailed report: `reports/rapportFL.pdf`
- Training results and visualizations: `results/`

## 🤝 Contributing

This project is a research implementation. For improvements or contributions:
1. Document changes clearly
2. Test on both datasets
3. Update results and comparisons


---

**Status**: Research Project
