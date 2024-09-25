# Network Anomaly Detection

## Overview
A machine learning project to detect anomalies in network traffic data.

## Project Structure
- **data/**: Contains raw and processed data.
- **notebooks/**: Jupyter Notebooks for exploratory data analysis.
- **src/**: Source code for data preprocessing, model training, and evaluation.
- **models/**: Trained machine learning models.
- **tests/**: Unit tests for the project.
- **scripts/**: Shell scripts to run various stages of the pipeline.

## Setup

### Prerequisites
- Python 3.8+
- Git

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/network-anomaly-detection.git
    cd network-anomaly-detection
    ```

2. Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Data Preprocessing
```bash
bash scripts/run_preprocessing.sh