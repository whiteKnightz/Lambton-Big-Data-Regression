from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt  # plotting
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import statsmodels.api as sm
import seaborn as sns

df = pd.read_csv('./data/measurements.csv')
df['distance'] = df['distance'].str.replace(',', '.').astype(float)
df['consume'] = df['consume'].str.replace(',', '.').astype(float)
df['temp_inside'] = df['temp_inside'].str.replace(',', '.').astype(float)

df.shape
print(df.shape)

df.info()
print(df.info())

df.head()
print(df.head())

df.describe()
print(df.describe())


sns.pairplot(df, x_vars=['distance', 'speed', 'temp_inside', 'gas_type'],
             y_vars='consume', height=4, aspect=1, kind='scatter')
plt.show()

sns.heatmap(df.corr(), cmap="YlGnBu", annot = True)
plt.show()


def regression_algorithm(X,y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7,
                                                        test_size=0.3, random_state=100)

    X_train_sm = sm.add_constant(X_train)

    lr = sm.OLS(y_train, X_train_sm).fit()
    lr.params
    print(lr.params)

    lr.summary()
    print(lr.summary())

    plt.scatter(X_train, y_train)
    plt.plot(X_train, 7.1164 + -0.1183 * X_train, 'r')
    plt.show()

    y_train_pred = lr.predict(X_train_sm)
    res = (y_train - y_train_pred)

    fig = plt.figure()
    sns.histplot(res, bins=15)
    plt.title('Error Terms', fontsize=15)
    plt.xlabel('y_train - y_train_pred', fontsize=15)
    plt.show()

    plt.scatter(X_train, res)
    plt.show()

    X_test_sm = sm.add_constant(X_test)
    y_test_pred = lr.predict(X_test_sm)
    y_test_pred
    print(y_test_pred)

    r_squared = r2_score(y_test, y_test_pred)
    r_squared
    print(r_squared)

    plt.scatter(X_test, y_test)
    plt.plot(X_test, y_test_pred, 'r')
    plt.show()

    X_train_lm, X_test_lm, y_train_lm, y_test_lm = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)

    X_train_lm.shape
    X_train_lm = X_train_lm.values.reshape(-1, 1)
    X_test_lm = X_test_lm.values.reshape(-1, 1)
    print(X_train_lm.shape)
    print(X_test_lm.shape)

    lm = LinearRegression()
    lm.fit(X_train_lm, y_train_lm)
    print("Intercept :", lm.intercept_)
    print('Slope :', lm.coef_)
    y_train_pred = lm.predict(X_train_lm)
    y_test_pred = lm.predict(X_test_lm)
    print(f'R2 score of Y train: {r2_score(y_train, y_train_pred)}')
    print(f'R2 score of Y test: {r2_score(y_test, y_test_pred)}')


dfSpeed30 = df.query('speed<30')
X = dfSpeed30['distance']
y = dfSpeed30['consume']
regression_algorithm(X, y)
