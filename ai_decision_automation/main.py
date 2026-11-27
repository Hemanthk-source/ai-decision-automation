"""
AI Decision Automation - simple base project
--------------------------------------------
Automates a decision (example: loan application) in 3 layers:

  1. Hard rules   -> instant approve/reject for policy-based cases
  2. ML model     -> predicts probability of approval for everything else
  3. Human review -> low-confidence cases are escalated, not auto-decided

Every decision is written to an audit log (decisions_log.jsonl).

Install:  pip install scikit-learn numpy pandas
Run:      python ai_decision_automation.py
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["age", "income", "credit_score", "loan_amount", "existing_debt"]
LOG_FILE = "decisions_log.jsonl"

# Confidence thresholds (tune these for your use case)
APPROVE_ABOVE = 0.75
REJECT_BELOW = 0.35


# ---------------------------------------------------------------- data + model
def generate_training_data(n=3000, seed=42):
    """Synthetic historical data. Replace with your real dataset (CSV/DB)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(18, 70, n),
        "income": rng.normal(50000, 18000, n).clip(10000),
        "credit_score": rng.normal(650, 90, n).clip(300, 850),
        "loan_amount": rng.normal(15000, 8000, n).clip(1000),
        "existing_debt": rng.normal(8000, 5000, n).clip(0),
    })
    score = (
        0.015 * (df.credit_score - 650)
        + 0.00004 * (df.income - 50000)
        - 0.00008 * (df.loan_amount - 15000)
        - 0.00010 * (df.existing_debt - 8000)
        + rng.normal(0, 0.5, n)
    )
    df["approved"] = (score > 0).astype(int)
    return df


def train_model(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df["approved"], test_size=0.2, random_state=1
    )
    model = make_pipeline(StandardScaler(), LogisticRegression())
    model.fit(X_train, y_train)
    print(f"[model] trained, test accuracy = {model.score(X_test, y_test):.2%}")
    return model


# ------------------------------------------------------------------ decisions
@dataclass
class Decision:
    timestamp: str
    applicant: dict
    decision: str        # APPROVED / REJECTED / MANUAL_REVIEW
    source: str          # rule / model
    confidence: float
    reason: str


def apply_rules(a):
    """Hard business rules. Return (decision, reason) or None to defer to the model."""
    if a["age"] < 18:
        return "REJECTED", "Applicant is under 18"
    if a["credit_score"] < 400:
        return "REJECTED", "Credit score below minimum policy (400)"
    if a["loan_amount"] > 10 * a["income"]:
        return "REJECTED", "Loan amount exceeds 10x annual income"
    return None


class DecisionEngine:
    def __init__(self, model):
        self.model = model

    def decide(self, applicant: dict) -> Decision:
        now = datetime.now().isoformat(timespec="seconds")

        # Layer 1: rules
        rule_result = apply_rules(applicant)
        if rule_result:
            d, reason = rule_result
            result = Decision(now, applicant, d, "rule", 1.0, reason)
        else:
            # Layer 2: ML model
            x = pd.DataFrame([applicant])[FEATURES]
            p = float(self.model.predict_proba(x)[0][1])
            if p >= APPROVE_ABOVE:
                result = Decision(now, applicant, "APPROVED", "model", p,
                                  f"High approval probability ({p:.0%})")
            elif p <= REJECT_BELOW:
                result = Decision(now, applicant, "REJECTED", "model", 1 - p,
                                  f"Low approval probability ({p:.0%})")
            else:
                # Layer 3: human in the loop
                result = Decision(now, applicant, "MANUAL_REVIEW", "model", p,
                                  f"Uncertain ({p:.0%}); needs human review")

        self._log(result)
        return result

    @staticmethod
    def _log(decision: Decision):
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(asdict(decision)) + "\n")


# ----------------------------------------------------------------------- demo
if __name__ == "__main__":
    model = train_model(generate_training_data())
    engine = DecisionEngine(model)

    samples = [
        {"age": 35, "income": 90000, "credit_score": 780, "loan_amount": 12000, "existing_debt": 3000},
        {"age": 22, "income": 20000, "credit_score": 520, "loan_amount": 25000, "existing_debt": 12000},
        {"age": 45, "income": 55000, "credit_score": 650, "loan_amount": 16000, "existing_debt": 8000},
        {"age": 17, "income": 30000, "credit_score": 700, "loan_amount": 5000, "existing_debt": 0},
        {"age": 40, "income": 30000, "credit_score": 700, "loan_amount": 400000, "existing_debt": 0},
    ]

    print("\n=== Decisions ===")
    for s in samples:
        r = engine.decide(s)
        print(f"{r.decision:<14} | via {r.source:<5} | conf {r.confidence:.0%} | {r.reason}")

    print(f"\nAudit log saved to {LOG_FILE}")
