def precision(
    y_true: list[int], y_predicted: list[tuple[int, float]], threshold: float
) -> tuple[float, int]:
    tp, fp = (0, 0)
    valid_entries: int = 0

    for label, prediction in zip(y_true, y_predicted):
        if prediction[1] > threshold:
            continue
        if label == prediction[0]:
            tp += 1
        else:
            fp += 1
        valid_entries += 1

    return tp / (tp + fp) if (tp + fp) > 0 else 0.0, valid_entries


def recall(
    y_true: list[int], y_predicted: list[list[tuple[int, float]]], threshold: float
) -> tuple[float, int]:
    tp, fn = (0, 0)

    for label, prediction in zip(y_true, y_predicted):
        print(label, prediction)
        predicted_labels = {
            predicted_label
            for predicted_label, distance in prediction
            if distance <= threshold
        }
        print(predicted_labels)

        if label in predicted_labels:
            tp += 1
        else:
            fn += 1

    return tp / (tp + fn) if (tp + fn) > 0 else 0.0, tp + fn
