# AI Decision Automation

A simple base project that automates decisions (example: loan approval) in three layers:

1. **Rules** - instant approve/reject for policy-based cases
2. **ML model** - logistic regression predicts approval probability
3. **Human review** - low-confidence cases are escalated

Every decision is saved to `decisions_log.jsonl` (audit log).

## Setup
    pip install -r requirements.txt

## Run
    python main.py

## Customize
- Replace `generate_training_data()` with your real dataset
- Edit `apply_rules()` for your business policies
- Tune `APPROVE_ABOVE` and `REJECT_BELOW` thresholds
