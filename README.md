# ComputingInnovationProject

## Requirements
Python3
pandas, numpy, scikit-learn, matplotlib, joblib

## How to run
### If you use venv
Ensure pandas, numpy, scikit-learn, matplotlib, joblib are installed

Run:
python3 model_execute.py

it should automatically run all scripts and create csv, joblib, pkl, etc. files.

### For Conda

Run:
conda env create -f environment.yml
conda activate COS30049
python3 model_execute.py

## Ensure these files a present
Ensure there is a basicDatasets folder in root otherwise it will fail.