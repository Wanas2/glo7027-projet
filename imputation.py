from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

def find_category_mappings(df, variable):
    return {k: i for i, k in enumerate(df[variable].dropna().unique(), 0)}


def integer_encode(df , variable, ordinal_mapping):
    df[variable] = df[variable].map(ordinal_mapping)


def knn_imputation(_df , cols, n_neighbors=1):
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
