import shap


class SHAPAnalyzer:
    def __init__(self, model, x_test):
        self.x_test = x_test
        self.explainer = shap.Explainer(model, self.x_test)
        self.shap_values = self.explainer.shap_values(self.x_test)

    def summary_plot(self):
        shap.summary_plot(self.shap_values, self.x_test)

    def dependence_plot(self, feature_name):
        print(self.x_test.columns)
        shap.dependence_plot(feature_name, self.shap_values, self.x_test)
