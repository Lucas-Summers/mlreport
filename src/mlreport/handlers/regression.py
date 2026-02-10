import numpy as np

from .base import ModelHandler


class RegressionHandler(ModelHandler):
    def metric_r2(self, splits: dict) -> dict:
        """R² Score"""
        results = {}
        for name, (X, y, y_pred) in splits.items():
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            results[name] = float(1 - (ss_res / ss_tot))
        return results

    def metric_mse(self, splits: dict) -> dict:
        """Mean Squared Error"""
        results = {}
        for name, (X, y, y_pred) in splits.items():
            results[name] = float(np.mean((y - y_pred) ** 2))
        return results

    def metric_rmse(self, splits: dict) -> dict:
        """Root Mean Squared Error"""
        results = {}
        for name, (X, y, y_pred) in splits.items():
            results[name] = float(np.sqrt(np.mean((y - y_pred) ** 2)))
        return results

    def metric_mae(self, splits: dict) -> dict:
        """Mean Absolute Error"""
        results = {}
        for name, (X, y, y_pred) in splits.items():
            results[name] = float(np.mean(np.abs(y - y_pred)))
        return results

    def metric_max_error(self, splits: dict) -> dict:
        """Max Error"""
        results = {}
        for name, (X, y, y_pred) in splits.items():
            results[name] = float(np.max(np.abs(y - y_pred)))
        return results

    def metric_median_ae(self, splits: dict) -> dict:
        """Median Absolute Error"""
        results = {}
        for name, (X, y, y_pred) in splits.items():
            results[name] = float(np.median(np.abs(y - y_pred)))
        return results

    def plot_predicted_vs_actual(self, ax, splits: dict):
        """Predicted vs Actual"""
        all_y = []

        for name, (X, y, y_pred) in splits.items():
            all_y.extend([y, y_pred])
            ax.scatter(
                y,
                y_pred,
                alpha=0.3,
                edgecolors="none",
                label=name.capitalize(),
            )

        min_val, max_val = np.concatenate(all_y).min(), np.concatenate(all_y).max()
        ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.7, label="Ideal")

        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.legend()

    def plot_residuals(self, ax, splits: dict):
        """Residuals vs Predicted"""
        for name, (X, y, y_pred) in splits.items():
            residuals = y - y_pred
            ax.scatter(
                y_pred,
                residuals,
                alpha=0.3,
                edgecolors="none",
                label=name.capitalize(),
            )

        ax.axhline(y=0, color="k", linestyle="--", alpha=0.7)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residuals")
        ax.legend()

    def plot_residual_hist(self, ax, splits: dict):
        """Residual Distribution"""
        for name, (X, y, y_pred) in splits.items():
            residuals = y - y_pred
            ax.hist(
                residuals,
                bins=30,
                edgecolor="black",
                alpha=0.5,
                label=name.capitalize(),
            )

        ax.axvline(x=0, color="k", linestyle="--", alpha=0.7)
        ax.set_xlabel("Residual")
        ax.set_ylabel("Frequency")
        ax.legend()

    def plot_qq(self, ax, splits: dict):
        """Q-Q Plot"""
        from scipy import stats

        first_split = next(iter(splits.values()))
        X, y, y_pred = first_split
        residuals = y - y_pred
        stats.probplot(residuals, dist="norm", plot=ax)
