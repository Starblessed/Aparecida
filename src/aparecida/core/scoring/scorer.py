from abc import ABC, abstractmethod


class Scorer(ABC):
    def __init__(self, threshold: float):
        self.threshold: float = threshold

    @abstractmethod
    def score(
        self, y_true: list[float], y_predicted: list[tuple[float, float]]
    ) -> tuple[float, int]: ...


class PrecisionScorer(Scorer):
    def score(
        self, y_true: list[float], y_predicted: list[tuple[float, float]]
    ) -> tuple[float, int]:
        tp, fp = (0, 0)
        valid_entries: int = 0

        for label, prediction in zip(y_true, y_predicted):
            if prediction[1] < self.threshold:
                continue
            if str(label) == str(prediction)[0]:
                tp += 1
            else:
                fp += 1
            valid_entries += 1

        return tp / (tp + fp) if (tp + fp) > 0 else 0.0, valid_entries
