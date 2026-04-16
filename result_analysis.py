from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

NO_REL = "NotValid"
file1 = "path/to/your/prediction/output/file"

def read_labels(file_path, sep="\t"):
    y_true = []
    y_pred = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            parts = line.strip().split(sep)

            gold = parts[0].strip()
            pred = parts[2].strip()

            y_true.append(gold)
            y_pred.append(pred)


    return y_true, y_pred

y_true, y_pred = read_labels(file1, sep="\t")


# define positive labels AFTER loading
positive_labels = sorted({l for l in y_true if l != NO_REL})

# Micro-F1 (all labels, all instances)  == accuracy for single-label multiclass
micro_f1_all = f1_score(y_true, y_pred, average="micro", zero_division=0)

# Macro over positive labels only (keep all instances; penalize FPs on none)
macro_f1_pos = f1_score(y_true, y_pred, labels=positive_labels, average="macro", zero_division=0)
macro_p_pos  = precision_score(y_true, y_pred, labels=positive_labels, average="macro", zero_division=0)
macro_r_pos  = recall_score(y_true, y_pred, labels=positive_labels, average="macro", zero_division=0)

print("Micro-F1 (all labels):", micro_f1_all)
print("Macro-F1 (exclude none):", macro_f1_pos)
print("Macro Precision (exclude none):", macro_p_pos)
print("Macro Recall (exclude none):", macro_r_pos)

print("\nPer-label report (ALL labels):")
print(classification_report(y_true, y_pred, digits=4, zero_division=0))

print("\nPer-label report (EXCLUDING none; macro/weighted computed over positives):")
print(classification_report(
    y_true, y_pred,
    labels=positive_labels,
    digits=4,
    zero_division=0
))