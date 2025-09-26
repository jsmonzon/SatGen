import deepdish as dd
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor  # you can use this to fit a model to data
from sklearn.model_selection import train_test_split #convient for splitting data
from sklearn.model_selection import cross_val_score

plt.style.use('../../../SatGen/notebooks/paper1/paper.mplstyle')
double_textwidth = 7.0 #inches
single_textwidth = 3.5 #inches


def gaussian_log_likelihood(y_pred, sigma_pred, y_true):
    """
    Compute the log-likelihood assuming Gaussian uncertainties.

    Parameters
    ----------
    y_pred : array_like
        Predicted values.
    sigma_pred : array_like
        Uncertainties (standard deviations) on the predicted values.
    y_true : array_like
        True values.

    Returns
    -------
    log_likelihood : float
        Total log-likelihood over all data points.
    """
    y_pred = np.asarray(y_pred)
    sigma_pred = np.asarray(sigma_pred)
    y_true = np.asarray(y_true)

    # Avoid division by zero or log(0)
    epsilon = 1e-10
    sigma_pred = np.clip(sigma_pred, epsilon, np.inf)

    # Gaussian log-likelihood
    logL = -0.5 * np.sum(
        np.log(2 * np.pi * sigma_pred**2) +
        ((y_true - y_pred)**2) / (sigma_pred**2)
    )

    return logL


class RunForestRun:

    def __init__(self, dataframe, target_key, **kwargs):
        self.dataframe = dataframe
        self.target_key = target_key

        # Default parameters with overrides via kwargs
        self.test_size = kwargs.get("test_size", 0.2)
        self.n_estimators = kwargs.get("n_estimators", 300)
        self.min_samples_leaf = kwargs.get("min_samples_leaf", 50)
        self.oob_score = kwargs.get("oob_score", False)
        self.cv = kwargs.get("cv", 5)
        self.verbose = kwargs.get("verbose", False)

        self.tt_split()
        self.train()
        self.r2_score()
        self.evaluate()

    def tt_split(self):
        self.inputs = self.dataframe.drop(columns=[self.target_key])
        self.target = self.dataframe[self.target_key]

        self.in_train, self.in_test, self.tar_train, self.tar_test = train_test_split(
            self.inputs, self.target, test_size=self.test_size
        )

    def train(self):
        self.RF_model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            oob_score=self.oob_score,
            random_state=42  # For reproducibility
        )
        self.RF_model.fit(self.in_train, self.tar_train)
        self.feature_importance_impur = self.RF_model.feature_importances_

        self.permuations = permutation_importance(self.RF_model, self.in_train, self.tar_train, scoring="explained_variance", n_repeats=10, random_state=42)
        self.feature_importance_perm = self.permuations.importances_mean
        self.features = pd.DataFrame(np.column_stack([self.inputs.keys(), self.feature_importance_impur, self.feature_importance_perm]), columns=["feature", "impurity", "permutation"])

    def r2_score(self):
        self.r2_scores = cross_val_score(
            self.RF_model, self.in_test, self.tar_test, scoring='explained_variance', cv=self.cv
        )
        self.r2_ave = np.mean(self.r2_scores)
        self.r2_std = np.std(self.r2_scores)

        if self.verbose:
            print(f"Mean R²: {self.r2_ave:.3f}")
            print(f"Std R²: {self.r2_std:.3f}")

    def evaluate(self):
        self.predictions = self.RF_model.predict(self.in_test)
        self.predictions_dist = np.array([tree.predict(self.in_test) for tree in self.RF_model.estimators_])
        self.predictions_std = np.std(self.predictions_dist, axis=0)
        self.residuals = self.tar_test - self.predictions

        self.logL = gaussian_log_likelihood(self.predictions,self.predictions_std,self.tar_test)
        self.chi2 = -2*self.logL

    def save_data(self, save_file):

        dictionary = {
            "target_test": self.tar_test,
            "target_prediction": self.predictions,
            "features": self.RF_model.feature_names_in_,
            "feature_importance": self.RF_model.feature_importances_,
            "R^2_score": self.r2_ave}

        dd.io.save(save_file, dictionary)

    def makeplot(self, savefile=None):

        fig, axes = plt.subplots(2, 2, gridspec_kw={'height_ratios': [2, 1]}, figsize=(7,5), sharex="col")

        sns.barplot(x = self.RF_model.feature_names_in_, y = self.feature_importance_impur, ax=axes[1,1], edgecolor="k", color="C0")
        axes[1,1].set_ylabel("relative importance")
        axes[1,1].set_ylim(0,1)

        axes[0,0].scatter(self.tar_test, self.predictions, marker=".", color="C0", s=2)
        sns.kdeplot(x=self.tar_test, y=self.predictions, ax=axes[0,0], levels=[1-0.99, 1-0.95, 1-0.68], color="grey", bw_adjust=1.5, linewidths=1)
        axes[0,0].set_xlabel("true " + self.target_key)
        axes[0,0].set_ylabel("predicted " + self.target_key)
        axes[0,0].axline((np.min(self.tar_test), np.min(self.tar_test)), slope=1, ls="--", color="k")

        axes[1,0].scatter(self.tar_test, self.residuals, marker=".", color="C0", s=2)
        sns.kdeplot(x=self.tar_test, y=self.residuals, ax=axes[1,0], levels=[1-0.99, 1-0.95, 1-0.68], color="grey", bw_adjust=1.5, linewidths=1)

        axes[1,0].axhline(0, ls="--", color="k")
        axes[1,0].set_ylabel("true - predicted")
        axes[1,0].set_xlabel("true " + self.target_key)

        axes[0,1].text(0.05, 0.05, f" N trees: {self.n_estimators} \n min leaf samples: {self.min_samples_leaf} \n R^2: {self.r2_ave:.2f} \n training size: {1 - self.test_size}",
                        transform=axes[0,1].transAxes, bbox=dict(facecolor='white', alpha=1, edgecolor="C0"))
        axes[0,1].axis("off")
        plt.tight_layout() 
        if savefile != None: 
            plt.savefig(savefile, bbox_inches="tight")
        plt.show()