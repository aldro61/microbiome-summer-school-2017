"""
Underfitting vs overfitting interactive example

Author: Alexandre Drouin
Inspired by http://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html

"""
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.widgets import Slider
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


MAX_DEGREE = 20

class DiscreteSlider(Slider):
    """
    A matplotlib slider widget with discrete steps.

    Source: https://stackoverflow.com/questions/13656387/can-i-make-matplotlib-sliders-more-discrete

    """
    def __init__(self, *args, **kwargs):
        """Identical to Slider.__init__, except for the "increment" kwarg.
        "increment" specifies the step size that the slider will be discritized
        to."""
        self.inc = kwargs.pop('increment', 0.5)
        Slider.__init__(self, *args, **kwargs)

    def set_val(self, val):
        discrete_val = int(val / self.inc) * self.inc
        # We can't just call Slider.set_val(self, discrete_val), because this
        # will prevent the slider from updating properly (it will get stuck at
        # the first step and not "slide"). Instead, we'll keep track of the
        # the continuous value as self.val and pass in the discrete value to
        # everything else.
        xy = self.poly.xy
        xy[2] = discrete_val, 1
        xy[3] = discrete_val, 0
        self.poly.xy = xy
        self.valtext.set_text(self.valfmt % discrete_val)
        if self.drawon:
            self.ax.figure.canvas.draw()
        self.val = val
        if not self.eventson:
            return
        for cid, func in self.observers.iteritems():
            func(discrete_val)


def fit_linear_regression(X, y, degree):
    return Pipeline([("polynomial_features", PolynomialFeatures(degree=degree,
                                                                include_bias=False)),
                     ("linear_regression", LinearRegression())]
                    ).fit(X, y)


class ModelSelectionPlot(object):
    def __init__(self, n_samples, random_state):
        self.inc = 1.0

        self.fig, (self.ax1, self.ax2) = plt.subplots(ncols=2)
        self.sliderax = self.fig.add_axes([0.2, 0.02, 0.6, 0.03], facecolor="lightgray")

        self.slider = DiscreteSlider(self.sliderax, 'Degree', 1, MAX_DEGREE,
                                     increment=self.inc, valinit=self.inc)
        self.slider.on_changed(self.update)
        self.slider.drawon = False

        # Generate training and testing data
        true_fun = lambda X: np.cos(1.5 * np.pi * X)
        X = random_state.rand(n_samples)
        y = true_fun(X) + random_state.randn(n_samples) * 0.1
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, train_size=0.5,
                                                                                random_state=random_state)
        self.X_train = self.X_train.reshape(-1, 1)
        self.X_test = self.X_test.reshape(-1, 1)

        # Plot the data
        self.ax1.plot(self.X_train, self.y_train, 'bo', markersize=5, label="Train")
        self.ax1.plot(self.X_test, self.y_test, 'ro', markersize=5, label="Test")
        x_draw_sampling = np.linspace(0, 1, 100)
        self.ax1.plot(x_draw_sampling, true_fun(x_draw_sampling), label="True function")

        # Train the learning algorithm using degree = 1
        estimator = fit_linear_regression(self.X_train, self.y_train, degree=1)
        x_draw_sampling = np.linspace(0, 1, 100).reshape(-1, 1)
        self.model_plot, = self.ax1.plot(x_draw_sampling, estimator.predict(x_draw_sampling), label="Model")

        # Plot the accuracy of the learned model
        self.train_score_plot, = self.ax2.plot([1], [estimator.score(self.X_train, self.y_train)], label="Training set", 
                                               markersize=5)
        self.test_score_plot, = self.ax2.plot([1], [estimator.score(self.X_test, self.y_test)], label="Testing set", 
                                              markersize=5)
        self.degree_marker = self.ax2.axvline(1, linestyle="--", color="red")

        # Left subplot formatting
        self.ax1.set_xlabel("X")
        self.ax1.set_ylabel("y")
        self.ax1.set_title("Model")
        self.ax1.legend()

        # Right subplot formatting
        self.ax2.set_xlabel("Degree hyperparameter")
        self.ax2.set_ylabel("Coefficient of determination ($r^2$)")
        self.ax2.set_xlim([1, MAX_DEGREE])
        self.ax2.set_ylim([0.5, 1])
        self.ax2.set_title("Accuracy")
        self.ax2.legend()

        # Main plot formatting
        plt.suptitle("Use the slider to explore different values of the degree hyperparameter")


    def update(self, degree):
        # Train the algorithm with the specified degree and plot its predictions
        estimator = fit_linear_regression(self.X_train, self.y_train, degree=int(degree))
        x_draw = np.linspace(0, 1, 100)
        self.model_plot.set_data(x_draw, estimator.predict(x_draw.reshape(-1, 1)))
        
        # Update the score plots
        def _update_score_plot(score_plot, new_score):
            t1, t2 = score_plot.get_data()
            t1 = np.hstack((t1, [degree]))
            t2 = np.hstack((t2, [new_score]))
            sorter = t1.argsort()
            t1 = t1[sorter]
            t2 = t2[sorter]
            score_plot.set_data(t1, t2)
        _update_score_plot(self.train_score_plot, estimator.score(self.X_train, self.y_train))
        _update_score_plot(self.test_score_plot, estimator.score(self.X_test, self.y_test))
        
        # Place the vertical marker at the current degree
        self.degree_marker.set_data([degree, degree], self.degree_marker.get_data()[1])
        
        # Update the slider's text and redraw the figure
        self.slider.valtext.set_text('{}'.format(degree))
        self.fig.canvas.draw()

    def show(self):
        plt.show()


if __name__ == "__main__":
    ModelSelectionPlot(n_samples=50, random_state=np.random.RandomState(1)).show()