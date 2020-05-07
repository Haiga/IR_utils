from toyTunning import l2rCodes
from sklearn.neural_network import MLPRegressor


#TODO CALL CORRECT READ DATASETS

X_train, y_train, query_id_train = []
X_test, y_test, query_id_test = []
dataset = "web10k"
metric = "ndcg"

###


#Params for tunning
paramCs = [1, 5, 10, 50, 100]
paramKernels = ['rbf', 'linear']

temp_bestndcg = 0
temp_bestqueries = []
temp_scoreTest = []
bestParams = [0, 0]

for kernel in paramKernels:
    for paramC in paramCs:
        model = MLPRegressor(hidden_layer_sizes=(100, ), verbose=False)
        model.fit(X_train, y_train)
        resScore = model.predict(X_test)

        scoreTest = [0] * len(y_test)
        c = 0
        for i in resScore:
            scoreTest[c] = i
            c = c + 1

        ndcg, queries = l2rCodes.getEvaluation(scoreTest, query_id_test, y_test, dataset, METRIC, "test")
        if ndcg > temp_bestndcg:
            temp_bestndcg = ndcg
            temp_bestqueries = queries
            temp_scoreTest = scoreTest
            bestParams[0] = kernel
            bestParams[1] = paramC

#TODO do something with bestParams