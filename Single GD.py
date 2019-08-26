import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

housing = pd.read_csv("data/Housing.csv")
print(housing.head())

housing["mainroad"] = housing["mainroad"].map({"yes":1, "no":0})
housing["guestroom"] = housing["guestroom"].map({"yes":1, "no":0})
housing['basement'] = housing['basement'].map({'yes': 1, 'no': 0})
housing['hotwaterheating'] = housing['hotwaterheating'].map({'yes': 1, 'no': 0})
housing['airconditioning'] = housing['airconditioning'].map({'yes': 1, 'no': 0})
housing['prefarea'] = housing['prefarea'].map({'yes': 1, 'no': 0})

status = pd.get_dummies(housing["furnishingstatus"], drop_first=True)
housing = pd.concat([housing, status], axis=1)
housing.drop(["furnishingstatus"], axis=1, inplace=True)
print(housing.head())

housing = (housing - housing.mean()) / housing.std()
print(housing.head())

x = housing["area"]
y = housing["price"]

sns.pairplot(housing, x_vars="area", y_vars="price", size=7, aspect=0.7, kind="scatter")
plt.show()

x = np.array(x)
y = np.array(y)


def gradient(x, y, m_current=0, c_current=0, iters = 1000, learing_rate = 0.01):
    N = float(len(y))
    gd_df = pd.DataFrame(columns=["m_current", "c_current", "cost"])
    for i in range(iters):
        y_current = (m_current * x) + c_current
        cost = sum([data**2 for data in (y-y_current)]) / N
        w_gradient =(-2/N) * sum(x*(y - y_current))
        d_gradient = (-2/N) * sum(y - y_current)
        m_current = m_current - (learing_rate * w_gradient)
        c_current = c_current - (learing_rate * d_gradient)
        gd_df.loc[i] = [m_current, c_current, cost]

    return(gd_df)

gradients = gradient(x, y)
print(gradients)

gradients.reset_index().plot.line(x='index', y=['cost'])
plt.show()