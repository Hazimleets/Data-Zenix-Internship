# Video Game Sales Forecasting (Enterprise-ready)


## Setup
1. Clone repo
2. Place `vgsales.csv` into `data/`
3. `python -m venv .venv && source .venv/bin/activate` (or Windows equivalent)
4. `pip install -r requirements.txt`


## Train
`python -m src.train --save models/rf_model.joblib`


## Evaluate
`python -m src.evaluate models/rf_model.joblib`


## Notes
- This project aggregates sales by Year and Platform; you can change aggregation granularity (e.g., monthly if you have timestamps).
- Use Optuna to tune hyperparameters; extend `train.py` to include an Optuna study objective wrapping scikit-learn estimators.