import pandas as pd
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('fake_data.csv')
print(df)

categorical_cols = ['state', 'fav_food']
numerical_cols = ['age']

feature_names = numerical_cols
for col in categorical_cols:
    dummies = pd.get_dummies(df[col], drop_first=True)
    print(dummies.columns.values)
    feature_names += list(dummies.columns.values)
    df = df.join(dummies)

print(df)

X = df[feature_names].values
print(feature_names)
print(X)
y = df.label.values
print(y)
# mapper = DataFrameMapper([
#         (['state'], OneHotEncoder()),
#         (['fav_food'], OneHotEncoder()),
#     ])

# X = mapper.fit_transform(df.copy())
# y = df.label

#print(X)