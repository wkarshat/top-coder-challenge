
"""
legacy_predictor.py

Interval/Type-based dispatch prediction for legacy reimbursement system.

USAGE:
    python legacy_predictor.py private_cases.json
    # Outputs to private_results.txt
"""

import sys
import pandas as pd
import numpy as np
import json

def predict_5day(days, miles, receipts):
    base = 100 * days
    mileage = 0.5 * miles if miles < 100 else 0.3 * miles + 20
    bonus = 100
    capped_receipts = min(receipts, 400)
    return round(base + mileage + capped_receipts + bonus, 2)

def predict_short_trip(days, miles, receipts):
    return round(120 * days + 0.7 * miles + min(receipts, 300), 2)

def predict_high_receipt(days, miles, receipts):
    return round(110 * days + 0.4 * miles + min(receipts, 350) - 50, 2)

def predict_baseline(days, miles, receipts):
    return round(100 * days + 0.5 * miles + min(receipts, 300), 2)

def predict_dispatch(row):
    days = int(row['trip_duration_days'])
    miles = float(row['miles_traveled'])
    receipts = float(row['total_receipts_amount'])
    if days == 5:
        return predict_5day(days, miles, receipts)
    elif miles < 100:
        return predict_short_trip(days, miles, receipts)
    elif receipts > 500:
        return predict_high_receipt(days, miles, receipts)
    else:
        return predict_baseline(days, miles, receipts)

def main(input_file):
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    elif input_file.endswith('.json'):
        with open(input_file, 'r') as f:
            data = json.load(f)
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            records = [case['input'] if 'input' in case else case for case in data]
            df = pd.DataFrame(records)
    else:
        print("Input file must be CSV or JSON.")
        return

    results = df.apply(predict_dispatch, axis=1)
    results.to_csv('private_results.txt', index=False, header=False, float_format='%.2f')
    print("Predictions written to private_results.txt")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python legacy_predictor.py private_cases.json")
    else:
        main(sys.argv[1])
