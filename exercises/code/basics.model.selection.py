"""
Underfitting vs overfitting interactive example

Author: Alexandre Drouin
Inspired by http://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html

"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from sklearn.model_selection import train_test_split


MAX_DEGREE = 30

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


class ChangingPlot(object):
    def __init__(self, n_samples, random_state):
        self.inc = 1.0

        self.fig, (self.ax1, self.ax2) = plt.subplots(ncols=2)
        self.sliderax = self.fig.add_axes([0.2, 0.02, 0.6, 0.03],
                                          axisbg='yellow')

        self.slider = DiscreteSlider(self.sliderax, 'Degree', 1, MAX_DEGREE,
                                     increment=self.inc, valinit=self.inc)
        self.slider.on_changed(self.update)
        self.slider.drawon = False

        true_fun = lambda X: np.cos(1.5 * np.pi * X)
        self.X = np.sort(random_state.rand(n_samples))
        self.y = true_fun(self.X) + random_state.randn(n_samples) * 0.1
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size=0.5,
                                                                                random_state=random_state)
        self.X_train = self.X_train.reshape(-1, 1)
        self.X_test = self.X_test.reshape(-1, 1)

        self.ax1.plot(self.X_train, self.y_train, 'bo', markersize=5, label="Train")
        self.ax1.plot(self.X_test, self.y_test, 'ro', markersize=5, label="Test")
        x_draw = np.linspace(0, 1, 100)
        self.ax1.plot(x_draw, true_fun(x_draw), label="True function")

        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        polynomial_features = PolynomialFeatures(degree=1,
                                                 include_bias=False)

        linear_regression = LinearRegression()
        pipeline = Pipeline([("polynomial_features", polynomial_features),
                             ("linear_regression", linear_regression)])
        pipeline.fit(self.X_train.reshape(-1, 1), self.y_train)
        x_draw = np.linspace(0, 1, 100).reshape(-1, 1)
        self.model, = self.ax1.plot(x_draw, pipeline.predict(x_draw), label="Model")

        self.training_risks, = self.ax2.plot([1], [pipeline.score(self.X_train, self.y_train)], label="Training set", markersize=5)
        self.testing_risks, = self.ax2.plot([1], [pipeline.score(self.X_test, self.y_test)], label="Testing set", markersize=5)
        self.vertical_degree = self.ax2.axvline(1, linestyle="--", color="red")

        # Left subplot formatting
        self.ax1.set_xlabel("X")
        self.ax1.set_ylabel("y")
        self.ax1.set_title("Model")
        self.ax1.legend()

        # Right subplot formatting
        self.ax2.set_xlabel("Degree")
        self.ax2.set_ylabel("Coefficient of determination ($r^2$)")
        self.ax2.set_xlim([0, MAX_DEGREE])
        self.ax2.set_ylim([0, 1])
        self.ax2.set_title("Accuracy")
        self.ax2.legend()

        plt.suptitle("Use the slider to explore different values of the degree hyperparameter")


    def update(self, value):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        polynomial_features = PolynomialFeatures(degree=int(value),
                                                 include_bias=False)

        linear_regression = LinearRegression()
        pipeline = Pipeline([("polynomial_features", polynomial_features),
                             ("linear_regression", linear_regression)])
        pipeline.fit(self.X_train.reshape(-1, 1), self.y_train)
        x_draw = np.linspace(0, 1, 100)
        self.model.set_data(x_draw, pipeline.predict(x_draw.reshape(-1, 1)))

        t1, t2 = self.training_risks.get_data()
        t1 = np.hstack((t1, [value]))
        t2 = np.hstack((t2, [pipeline.score(self.X_train, self.y_train)]))
        sorter = t1.argsort()
        t1 = t1[sorter]
        t2 = t2[sorter]
        self.training_risks.set_data(t1, t2)

        t1, t2 = self.testing_risks.get_data()
        t1 = np.hstack((t1, [value]))
        t2 = np.hstack((t2, [pipeline.score(self.X_test, self.y_test)]))
        sorter = t1.argsort()
        t1 = t1[sorter]
        t2 = t2[sorter]
        self.testing_risks.set_data(t1, t2)

        self.vertical_degree.set_data([value, value], self.vertical_degree.get_data()[1])

        self.slider.valtext.set_text('{}'.format(value))
        self.fig.canvas.draw()

    def show(self):
        plt.show()

p = ChangingPlot(50, np.random.RandomState(1))
p.show()