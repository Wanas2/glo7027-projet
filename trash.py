# for n in n_estimators:
#     for d in max_depth:
#         print("n_estimators = %d and max_depth = %d" % (n, d))

#         result = dict()
#         columns = ['fit_time', 'score_time', 'test_accuracy', 'train_accuracy', 'test_precision_weighted', 'train_precision_weighted', 'test_recall_weighted', 'train_recall_weighted']
#         for col in columns:
#             result[col] = list()
        
#         pipe = make_pipeline(column_trans, RandomForestClassifier(n_estimators=n, max_depth=d, random_state=0))

#         for X_imputed in imputed_data:
#             result_cv = cross_validate(pipe, X_imputed, y, cv=5, scoring=scoring, return_train_score=True)
#             for col in columns:
#                 if col is 'fit_time' or col is 'score_time':
#                     result[col].append(np.sum(result_cv[col]))
#                 else:
#                     result[col].append(np.mean(result_cv[col]))
        
#         df = pd.DataFrame.from_dict(result)           
#         df.index = ["n_neighbors=%d" % n for n in [1, 2, 3, 5, 7, 11]]
#         display(df)

#         print("\n")
            

# for n in n_estimators:
#     result = dict()
#     columns = ['fit_time', 'score_time', 'test_accuracy', 'train_accuracy', 'test_precision_weighted', 'train_precision_weighted', 'test_recall_weighted', 'train_recall_weighted']
#     for col in columns:
#         result[col] = list()

#     for d in max_depth:        
#         pipe = make_pipeline(column_trans, RandomForestClassifier(n_estimators=n, max_depth=d, random_state=0))

#         result_cv = cross_validate(pipe, X, y, cv=5, scoring=scoring, return_train_score=True)
#         for col in columns:
#             if col is 'fit_time' or col is 'score_time':
#                 result[col].append(np.sum(result_cv[col]))
#             else:
#                 result[col].append(100 * np.mean(result_cv[col]))

#     plt.figure(figsize=(30, 15))

#     plt.subplot(121)
#     plt.scatter(list(max_depth), result['test_accuracy'], label="Taux de succès")
#     plt.scatter(list(max_depth), result['test_precision_weighted'], label="Précision")
#     plt.scatter(list(max_depth), result['test_recall_weighted'], label="Rappel")

#     plt.xlabel("Profondeur maximale")
#     plt.ylabel("Performances")

#     plt.title("Performances (en pourcentages) pour n_estimators = %d" % n)
#     plt.legend()

#     plt.subplot(122)
#     plt.scatter(list(max_depth), result['fit_time'], label="Temps d'entrainement")
#     plt.scatter(list(max_depth), result['score_time'], label="Temps de test de performance")

#     plt.xlabel("Profondeur maximale")
#     plt.ylabel("Temps")

#     plt.title("Temps (en secondes) pour n_estimators = %d" % n)G
#     plt.legend()
    
#     plt.savefig("img/RandomForest %d" % d)
#     plt.show()