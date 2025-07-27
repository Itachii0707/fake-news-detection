from __future__ import annotations

import argparse
import joblib
import numpy as np


def predict_single(model_path, text: str):
    bundle = joblib.load(model_path)
    pipe = bundle["pipeline"]
    label_encoder = bundle["label_encoder"]
    label_names = bundle["label_names"]

    proba_supported = hasattr(pipe, "predict_proba")

    pred_idx = pipe.predict([text])[0]
    pred_label = label_encoder.inverse_transform([pred_idx])[0]

    if proba_supported:
        probs = pipe.predict_proba([text])[0]
        # Align class probs with label_encoder classes
        # If scikit-learn changes order, this ensures we print correct mapping.
        class_indices = np.arange(len(label_names))
        out = {label_names[i]: float(probs[i]) for i in class_indices}
    else:
        out = None

    return pred_label, out


def main():
    parser = argparse.ArgumentParser(description="Infer on a single text with a trained model.")
    parser.add_argument("--model", type=str, required=True, help="Path to the saved joblib model")
    parser.add_argument("--text", type=str, required=True, help="The text/article to classify")
    args = parser.parse_args()

    label, probs = predict_single(args.model, args.text)

    print(f"Predicted label: {label}")
    if probs is not None:
        print("Class probabilities:")
        for k, v in probs.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
