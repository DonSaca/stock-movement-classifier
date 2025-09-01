# Stock Movement Classifier (LSTM vs Transformer)

This project compares **LSTM** and **Transformer** models for stock price movement prediction.  
The focus is on building a clean, well-structured machine learning pipeline that can be run and reproduced on different machines.

## Project Structure
- `src/` – Python source code
- `data/` – Raw and processed data (ignored by Git)
- `notebooks/` – Jupyter notebooks for experiments
- `app/` – Application layer (UI, deployment code)
- `.venv/` – Local Python virtual environment (ignored by Git)

## Setup
1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1   # on Windows PowerShell
3.
pip install -r requirements.txt