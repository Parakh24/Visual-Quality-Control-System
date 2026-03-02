# Visual-Quality-Control-System
## Dataset
Dataset link: https://drive.google.com/drive/folders/1KA5_0-u9NVzR5i_zlL2UOhSkkcrt2-5n?usp=sharing

📸 Visual Quality Control System

An end-to-end Visual Quality Control System for automated defect detection and inspection in manufacturing/production workflows. This project leverages deep learning and computer vision to identify visual anomalies and ensure high-quality outputs, reducing manual inspection effort and improving consistency.

🧠 Overview

The Visual Quality Control System is designed to analyze visual data (images) from manufacturing processes, perform feature extraction and classification, and provide actionable insights on product quality. The system supports training custom models as well as inference for automated inspection.

🔗 Dataset:
Google Drive link (contains training/testing images and annotations):
https://drive.google.com/drive/folders/1KA5_0-u9NVzR5i_zlL2UOhSkkcrt2-5n?usp=sharing

📌 Features

📷 Visual anomaly detection using deep learning

🧠 Model training & evaluation utilities

🔍 Detailed results and metrics logging

📊 Sample outputs and visualizations included

🧪 Support for experimentation and rapid prototyping

📂 Project Structure
Visual-Quality-Control-System/
├── assets/                     # Sample outputs & visualization assets
├── data/                       # Dataset and annotation files
├── experiments/                # Experiment results & logs
├── logs/                       # Training & evaluation logs
├── models/trained/             # Saved trained models
├── results/                    # Inference/validation outputs
├── src/                        # Source code modules
├── .gitignore
├── benchmark_cpu_latency.py    # CPU performance benchmark utility
├── requirements.txt            # Python dependencies
└── run_week3_analysis.py       # Example experiment runner
🛠️ Tech Stack

Language: Python

Libraries & Tools:

TensorFlow / PyTorch (for deep learning)

OpenCV (image processing)

NumPy, Pandas (data handling)

Matplotlib / Seaborn (visualization)

Environment: Compatible with Python 3.7+ and common ML workflows

🚀 Installation

Clone the repository

git clone https://github.com/Parakh24/Visual-Quality-Control-System.git
cd Visual-Quality-Control-System

Set up a virtual environment

python3 -m venv venv
source venv/bin/activate

Install dependencies

pip install -r requirements.txt
📈 Usage
🧪 Training

Train a model on the dataset using provided training scripts in the src/ folder.

Example:

python src/train_model.py \
  --data_dir data/ \
  --output_dir models/trained/
📊 Evaluation & Inference

Run evaluation or inference using saved models:

python src/inference.py \
  --model_path models/trained/latest_model.pth \
  --input_dir data/test/
📌 Benchmarking

To benchmark CPU performance:

python benchmark_cpu_latency.py
📊 Results

Sample model outputs and performance graphs are available under the assets/ directory. These include intermediate training graphs and inference visuals for qualitative inspection.

🙌 Contributing

Contributions to improve model architectures, expand dataset support, or optimize performance are welcome!
Before submitting a pull request:

Fork the project

Create a feature branch (git checkout -b feature/xyz)

Add enhancements or fixes

Submit a PR with a clear description

📜 License

This repository is licensed under the Apache-2.0 License.
