# Rouler ce .py pour avoir les datasets utilisés dans naiveBayes.py
# ou rouler Modif_donnee_CNB_train_test_NoOrdinal.py

#%%
# Importations
from re import A
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression

all_data_train = pd.read_csv(r"data_train.csv", low_memory=False)
all_data_test = pd.read_csv(r"data_test.csv", low_memory=False)

#Commenter pour utiliser imputation knn
#imp_data_train = pd.read_csv(r"final_train_imputed_2.csv")
#imp_data_test = pd.read_csv(r"final_test_imputed_2.csv")

#Commenter pour utiliser imputation logistique
#Changer le chiffre pour choisir combien de voisin
#Les datasets viennent de main.py pour les données test et les données train
#  On s'en sert seulement pour avoir cps19_imp_iss_party
# n=7
imp_data_train = pd.read_csv("data_train_imputed.csv")
imp_data_test = pd.read_csv("data_test_imputed.csv")

all_data_train = all_data_train.drop(["Unnamed: 0", "Unnamed: 0.1"], axis=1)
all_data_test = all_data_test.drop(["Unnamed: 0", "Unnamed: 0.1"], axis=1)

#Attribut initiaux pour référence
attr_train = ['cps19_fed_gov_sat', 'cps19_spend_env', 'cps19_fed_id', 'cps19_province', 'cps19_education', 'cps19_age', 'votechoice']

#Mettre les attributs choisi
# attr_train = ['cps19_fed_gov_sat', 'cps19_spend_env', 'cps19_age', 'votechoice']
attr_test = attr_train.copy()
attr_test.pop()

#Sans cps19_imp_iss_party
#data_train = all_data_train[attr_train]
#data_test = all_data_test[attr_test]

#Sans Imputé
data_train = pd.concat([all_data_train["cps19_imp_iss_party"], all_data_train[attr_train]], axis=1)
data_test = pd.concat([all_data_test["cps19_imp_iss_party"], all_data_test[attr_test]], axis=1)

#Avec imputé
# data_train = pd.concat([imp_data_train["cps19_imp_iss_party"], all_data_train[attr_train]], axis=1)
# data_test = pd.concat([imp_data_test["cps19_imp_iss_party"], all_data_test[attr_test]], axis=1)


# party rating
party_rating = ["cps19_party_rating_23","cps19_party_rating_24","cps19_party_rating_25","cps19_party_rating_26","cps19_party_rating_27","cps19_party_rating_28"]
data_train = pd.concat([data_train, all_data_train[party_rating].fillna(50)], axis =1)
data_test = pd.concat([data_test, all_data_test[party_rating].fillna(50)], axis =1)


#%%
# On catégorise les party_rating pour éviter des probabilités trop près de 0
# Nécessaire pour tout attributs numériques
ratingCat = ["10", "20", "30", "40", "50", "60", "70", "80", "90", "100"]
for party_r in party_rating:
    category_train = pd.cut(data_train[party_r],bins=[-1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],labels=ratingCat)
    data_train[party_r] = category_train

    category_test = pd.cut(data_test[party_r],bins=[-1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],labels=ratingCat)
    data_test[party_r] = category_test

    # On modifie en OrdinalEncoder car ce sont des catégories ordinales
    le = OrdinalEncoder(categories=[ratingCat])
    train_le = le.fit_transform(data_train.loc[:,[party_r]])
    train_le = pd.DataFrame(train_le, columns=[party_r])
    data_train = pd.concat([train_le, data_train.drop([party_r], axis=1)], axis=1)

    le = OrdinalEncoder(categories=[ratingCat])
    test_le = le.fit_transform(data_test.loc[:,[party_r]])
    test_le = pd.DataFrame(test_le, columns=[party_r])
    data_test = pd.concat([test_le, data_test.drop([party_r], axis=1)], axis=1)

#%%
# On catégorise l'âge pour éviter des probabilités trop près de 0
# Nécessaire pour tout attributs numériques
ageCat = ["18-24", "25-34", "35-49", "50-59", "60-69", "70-79", "80-89", "90-99"]

category_train = pd.cut(data_train["cps19_age"],bins=[17, 24, 34, 49, 59, 69, 79, 89, 99],labels=ageCat)
data_train["cps19_age"] = category_train

category_test = pd.cut(data_test["cps19_age"],bins=[17, 24, 34, 49, 59, 69, 79, 89, 99],labels=ageCat)
data_test["cps19_age"] = category_test

# On modifie cps19_age en OrdinalEncoder car ce sont des catégories ordinales
le = OrdinalEncoder(categories=[ageCat])
train_le = le.fit_transform(data_train.loc[:,["cps19_age"]])
train_le = pd.DataFrame(train_le, columns=["cps19_age"])
modifiedData_train = pd.concat([train_le, data_train.drop(["cps19_age"], axis=1)], axis=1)

le = OrdinalEncoder(categories=[ageCat])
test_le = le.fit_transform(data_test.loc[:,["cps19_age"]])
test_le = pd.DataFrame(test_le, columns=["cps19_age"])
modifiedData_test = pd.concat([test_le, data_test.drop(["cps19_age"], axis=1)], axis=1)


#%%
# On modifie cps19_fed_gov_sat en OrdinalEncoder car ce sont des catégories ordinales
le = OrdinalEncoder(categories=[["Very satisfied", "Fairly satisfied", "Not very satisfied", "Not at all satisfied", "Don't know/ Prefer not to answer"]])
fed_sat_train_le = le.fit_transform(data_train.loc[:,["cps19_fed_gov_sat"]])
train_le = pd.DataFrame(fed_sat_train_le, columns=["cps19_fed_gov_sat"])
modifiedData_train = pd.concat([train_le, modifiedData_train.drop(["cps19_fed_gov_sat"], axis=1)], axis=1)

le = OrdinalEncoder(categories=[["Very satisfied", "Fairly satisfied", "Not very satisfied", "Not at all satisfied", "Don't know/ Prefer not to answer"]])
fed_sat_test_le = le.fit_transform(data_test.loc[:,["cps19_fed_gov_sat"]])
test_le = pd.DataFrame(fed_sat_test_le, columns=["cps19_fed_gov_sat"])
modifiedData_test = pd.concat([test_le, modifiedData_test.drop(["cps19_fed_gov_sat"], axis=1)], axis=1)


#%%
# On modifie cps19_spend_env en OrdinalEncoder car ce sont des catégories ordinales
le = OrdinalEncoder(categories=[["Spend less", "Spend about the same as now", "Spend more", "Don't know/ Prefer not to answer"]])
spend_env_train_le = le.fit_transform(modifiedData_train.loc[:,["cps19_spend_env"]])
train_le = pd.DataFrame(spend_env_train_le, columns=["cps19_spend_env"])
modifiedData_train = pd.concat([train_le, modifiedData_train.drop(["cps19_spend_env"], axis=1)], axis=1)

le = OrdinalEncoder(categories=[["Spend less", "Spend about the same as now", "Spend more", "Don't know/ Prefer not to answer"]])
spend_env_test_le = le.fit_transform(modifiedData_test.loc[:,["cps19_spend_env"]])
test_le = pd.DataFrame(spend_env_test_le, columns=["cps19_spend_env"])
modifiedData_test = pd.concat([test_le, modifiedData_test.drop(["cps19_spend_env"], axis=1)], axis=1)


# %%
# On modifie en OneHotEncoder
ohe_attr = ["cps19_imp_iss_party", 'cps19_fed_id', 'cps19_province', 'cps19_education']
not_ohe_attr = ["cps19_age", "cps19_party_rating_23","cps19_party_rating_24","cps19_party_rating_25","cps19_party_rating_26","cps19_party_rating_27","cps19_party_rating_28","cps19_spend_env", "cps19_fed_gov_sat", "votechoice"]
ohe = OneHotEncoder()
array_train_ohe = ohe.fit_transform(modifiedData_train.drop(not_ohe_attr, axis=1)).toarray()
train_ohe = pd.DataFrame(array_train_ohe)
modifiedData_train = pd.concat([train_ohe, modifiedData_train[not_ohe_attr]], axis=1)

ohe = OneHotEncoder()
array_test_ohe = ohe.fit_transform(modifiedData_test.drop(not_ohe_attr[:-1], axis=1)).toarray()
test_ohe = pd.DataFrame(array_test_ohe)
modifiedData_test = pd.concat([test_ohe, modifiedData_test[not_ohe_attr[:-1]]], axis=1)



# %%
modifiedData_train.to_csv("modifiedData.csv", index=False)
modifiedData_test.to_csv("modifiedDataTest.csv", index=False)

modifiedData_train

# %%
