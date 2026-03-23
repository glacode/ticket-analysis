# ServiceNow Ticket Semantic Analysis

An AI-powered tool for clustering and analyzing IT tickets based on semantic similarity. It uses modern Natural Language Processing (NLP) to group tickets with similar topics, helping IT managers identify recurring issues and automate classification.

## 🚀 Features

- **Semantic Embeddings**: Uses `sentence-transformers` (Multilingual MiniLM) to understand the meaning behind ticket descriptions.
- **Dimensionality Reduction**: Employs **UMAP** for efficient high-dimensional data processing.
- **Clustering**: Groups tickets using **K-Means** with human-readable labels generated from the most representative ticket in each cluster.
- **Interactive Visualization**: Generates a **Sunburst Chart** (HTML) using Plotly for hierarchical exploration.
- **Detailed Reports**: Creates a standalone HTML report with collapsible sections for each cluster.

## 🛠️ Technology Stack

- **Python 3.10+**
- **Pandas** for data manipulation.
- **Scikit-Learn** for clustering.
- **PyTorch (CPU-only)** for NLP model inference.
- **Plotly** for interactive charts.
- **UMAP-learn** for manifold learning.

## 📦 Installation

### 1. Prerequisites
Ensure you have Python 3.10 installed on your system.

### 2. Set up the Environment
Create a virtual environment and install the dependencies:

```bash
# Create the virtual environment
python3.10 -m venv .venv

# Activate it
source .venv/bin/activate

# Install CPU-only PyTorch first (to save disk space)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
pip install -r requirements.txt
```

## 📖 Usage

1. Prepare your input file: `servicenow_tickets.csv`. It should contain at least the following columns: `number`, `short_description`, and `description`.
2. Run the analysis script:

```bash
python ticket_analysis.py
```

## 📊 Outputs

- **`semantic_clustered_tickets.csv`**: The original data with added `cluster_id` and `cluster_label` columns.
- **`semantic_report_pro.html`**: An interactive Sunburst chart to visualize ticket distribution.
- **`cluster_details.html`**: A detailed report containing all tickets grouped by their semantic category.

## 📄 License
This project is open-source and available for any personal or commercial use.
