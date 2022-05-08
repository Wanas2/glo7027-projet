# %%
import numpy as np
import pandas as pd

# %%
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# %%
data = pd.read_csv("data_train.csv")

# %%
columns = ['cps19_fed_gov_sat', 'cps19_spend_env', 'cps19_fed_id', 'cps19_citizenship', 'cps19_province', 'cps19_education', 'cps19_age', 'cps19_imp_iss_party']

# %%
X = data.copy()[columns]
X.isna().sum()

# %%
y = data.copy()["votechoice"]
y.isna().sum()

# %%
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

# %%
def find_category_mappings(df, variable):
    return {k: i for i, k in enumerate(df[variable].dropna().unique(), 0)}


def integer_encode(df , variable, ordinal_mapping):
    df[variable] = df[variable].map(ordinal_mapping)


def imputation(_df , cols, n_neighbors=1):
    mm = MinMaxScaler()
    _mapping = dict()
    
    df = _df.copy()
    for variable in cols:
        mappings = find_category_mappings(df, variable)
        _mapping[variable] = mappings

    for variable in cols:
        integer_encode(df, variable, _mapping[variable])  

    sca = mm.fit_transform(df)
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    knn = knn_imputer.fit_transform(sca)
    df.iloc[:,:] = mm.inverse_transform(knn)
    for i in df.columns : 
        df[i] = round(df[i]).astype('int')

    for i in cols:
        inv_map = {v: k for k, v in _mapping[i].items()}
        df[i] = df[i].map(inv_map)
    return df

# %%
imputed_data = []
for n in [1, 2, 3, 5, 7, 11]:
    imputed_data.append(imputation(X, columns, n_neighbors=n))

# %%
from IPython.display import display

# %%
for _X in imputed_data:
    display(_X)

# %%
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

# %%
column_trans = make_column_transformer(
                    (OneHotEncoder(), list(X.select_dtypes('object').columns)), 
                    remainder='passthrough'
               )

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

# %%
n_estimators = range(5, 11)
max_depth = range(5, 15)

# %%
from sklearn.model_selection import cross_validate

# %%
import warnings
warnings.filterwarnings("ignore")

# %%
scoring = ['accuracy', 'precision_weighted', 'recall_weighted']

# %%
for n in n_estimators:
    for d in max_depth:
        print("n_estimators = %d and max_depth = %d" % (n, d))

        result = dict()
        columns = ['fit_time', 'score_time', 'test_accuracy', 'train_accuracy', 'test_precision_weighted', 'train_precision_weighted', 'test_recall_weighted', 'train_recall_weighted']
        for col in columns:
            result[col] = list()
        
        pipe = make_pipeline(column_trans, RandomForestClassifier(n_estimators=n, max_depth=d, random_state=0))

        for X_imputed in imputed_data:
            result_cv = cross_validate(pipe, X_imputed, y, cv=5, scoring=scoring, return_train_score=True)
            for col in columns:
                if col is 'fit_time' or col is 'score_time':
                    result[col].append(np.sum(result_cv[col]))
                else:
                    result[col].append(np.mean(result_cv[col]))
        
        df = pd.DataFrame.from_dict(result)           
        df.index = ["n_neighbors=%d" % n for n in [1, 2, 3, 5, 7, 11]]
        display(df)

        print("\n")
            


