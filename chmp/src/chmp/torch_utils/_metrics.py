from typing import Optional


def ensure_metric(obj):
    return obj if isinstance(obj, Metric) else AveragedMetric(obj)


class Metric:
    value_class: Optional[type] = None

    def zero(self):
        return self.value_class()

    def compute(self, pred, y):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError


class AveragedMetricValue:
    def __init__(self, value=0.0, count=0):
        self.value = value
        self.count = count

    def __add__(self, other):
        return type(self)(self.value + other.value, self.count + other.count)

    def get(self):
        return self.value / self.count


class AveragedMetric(Metric):
    value_class = AveragedMetricValue

    def __init__(self, metric):
        self.metric = metric

    def name(self):
        try:
            return self.metric.__name__

        except AttributeError:
            return str(self.metric)

    def compute(self, pred, y):
        return AveragedMetricValue(len(y) * float(self.metric(pred, y)), len(y))
