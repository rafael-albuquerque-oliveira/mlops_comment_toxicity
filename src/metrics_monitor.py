from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy

class MetricsMonitor:
    def __init__(self):
        """Initialize metrics for monitoring."""
        self.precision_metric = Precision()
        self.recall_metric = Recall()
        self.accuracy_metric = CategoricalAccuracy()

    def update_metrics(self, y_true, y_pred):
        """Update metrics based on true and predicted values."""
        self.precision_metric.update_state(y_true.flatten(), y_pred.flatten())
        self.recall_metric.update_state(y_true.flatten(), y_pred.flatten())
        self.accuracy_metric.update_state(y_true.flatten(), y_pred.flatten())

    def get_results(self):
        """Return calculated metrics results."""
        return {
             'Precision': self.precision_metric.result().numpy(),
             'Recall': self.recall_metric.result().numpy(),
             'Accuracy': self.accuracy_metric.result().numpy()
         }
