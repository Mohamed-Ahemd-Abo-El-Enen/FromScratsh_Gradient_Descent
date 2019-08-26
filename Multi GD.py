import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

housing = pd.read_csv('Housing.csv')
housing.head()

housing['mainroad'] = housing['mainroad'].map({'yes': 1, 'no': 0})
housing['guestroom'] = housing['guestroom'].map({'yes': 1, 'no': 0})
housing['basement'] = housing['basement'].map({'yes': 1, 'no': 0})
housing['hotwaterheating'] = housing['hotwaterheating'].map({'yes': 1, 'no': 0})
housing['airconditioning'] = housing['airconditioning'].map({'yes': 1, 'no': 0})
housing['prefarea'] = housing['prefarea'].map({'yes': 1, 'no': 0})

status = pd.get_dummies(housing['furnishingstatus'], drop_first=True)
housing = pd.concat([housing, status], axis=1)
housing.drop(['furnishingstatus'], axis=1, inplace=True)

print(housing.head())

housing = (housing - housing.mean())/housing.std()
print(housing.head())

x = housing[["area", "bedrooms"]]
y = housing["price"]

x["intercept"] = 1
x = x.reindex_axis(['intercept','area','bedrooms'], axis=1)
print(x.head())

x = np.array(x)
y = np.array(y)

theta = np.matrix(np.array([0, 0, 0]))
alpha = 0.01
iterations = 1000


def compute_cost(x, y, theta):
    return np.sum(np.square(np.matmul(x, theta) - y)) / (2*len(y))


def gradient_descent_multi(x, y, alpha, iterations):
    theta = np.zeros(x.shape[1])
    m = len(x)
    gdm_df = pd.DataFrame(columns=["bets", "cost"])

    for i in range(iterations):
        gradient = (1/m) * np.matmul(x.T, np.matmul(x, theta) - y)
        theta = theta - alpha * gradient
        cost = compute_cost(x, y, theta)
        gdm_df.loc[i] = [theta, cost]

    return gdm_df

print(gradient_descent_multi(x, y, alpha, iterations))
gradient_descent_multi(x, y, alpha, iterations).reset_index().plot.line(x="index", y=["cost"])
plt.show()