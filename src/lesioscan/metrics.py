from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score  # Memuat sklearn.metrics untuk fungsi yang digunakan pada file ini.

def evaluate(y_true, y_pred, labels):  # Menghitung metrik klasifikasi dan confusion matrix.
    ids = list(range(len(labels)))  # Menetapkan indeks kelas.
    report = classification_report(y_true, y_pred, labels=ids, target_names=labels, output_dict=True, zero_division=0)  # Menetapkan laporan per kelas.
    return {'accuracy': float(accuracy_score(y_true, y_pred)), 'macro_f1': float(f1_score(y_true, y_pred, labels=ids, average='macro', zero_division=0)), 'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)), 'per_class': report, 'confusion_matrix': confusion_matrix(y_true, y_pred, labels=ids).tolist()}  # Mengembalikan hasil fungsi kepada pemanggil.
