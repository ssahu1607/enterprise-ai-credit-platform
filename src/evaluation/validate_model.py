import mlflow
from mlflow import MlflowClient


MODEL_NAME = "credit-risk-model"
MODEL_VERSION = "2"

# Demo validation thresholds
MIN_RECALL = 0.85
MIN_F1 = 0.80
MIN_PRECISION = 0.70
MIN_ROC_AUC = 0.65


client = MlflowClient()

# Get Version 2 information
model_version = client.get_model_version(
    name=MODEL_NAME,
    version=MODEL_VERSION
)

run_id = model_version.run_id

# Get metrics from the training run
run = client.get_run(run_id)
metrics = run.data.metrics

print("\nModel Validation")
print("----------------")
print(f"Model   : {MODEL_NAME}")
print(f"Version : {MODEL_VERSION}")
print(f"Run ID  : {run_id}")

print("\nMetrics:")
print(f"Recall    : {metrics['recall']:.3f}")
print(f"F1 Score  : {metrics['f1_score']:.3f}")
print(f"Precision : {metrics['precision']:.3f}")
print(f"ROC-AUC   : {metrics['roc_auc']:.3f}")


# Validation checks
checks = {
    "Recall": metrics["recall"] >= MIN_RECALL,
    "F1 Score": metrics["f1_score"] >= MIN_F1,
    "Precision": metrics["precision"] >= MIN_PRECISION,
    "ROC-AUC": metrics["roc_auc"] >= MIN_ROC_AUC,
}


print("\nValidation Checks:")

all_passed = True

for metric, passed in checks.items():

    if passed:
        print(f"PASS - {metric}")
    else:
        print(f"FAIL - {metric}")
        all_passed = False


if all_passed:
    print("\nMODEL VALIDATION PASSED")
    print("Model is eligible for approval.")
else:
    print("\nMODEL VALIDATION FAILED")
    print("Model must not be promoted.")