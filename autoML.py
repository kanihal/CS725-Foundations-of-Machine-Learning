# AUTO REGRESSOR

# INSTRUCTIONS
# 1. USE camelCase names for variable as well function names

##################################################################################################################################

# importing related packages
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn import preprocessing as pre
from sklearn import feature_extraction as fe
from sklearn import datasets
import pickle
import copy
import time

# make it 1, while debugging else make it 0
debug = 0


##################################################################################################################################
class AutoRegressor(object):
    # Constructor
    def __init__(self, trainCsvFile, testCsvFile, yLabel):
        if (trainCsvFile):
            # load input csv file data
            df = pd.read_csv(trainCsvFile)
            self.yLabel = yLabel
            self.x = (df.loc[:, df.columns != self.yLabel])
            self.y = (df.loc[:, self.yLabel])
        else:
            # use sklearn datasets - we can comment this afterwards
            dataset = datasets.load_diabetes()
            self.x = dataset.data
            self.y = dataset.target

        # input data for which we need to predict the output
        if testCsvFile:
            df = pd.read_csv(testCsvFile)
            self.xTest = df
        else:  # for time being
            self.xTest = None

    # To save objects
    def saveObject(self, obj, filename):
        with open(filename, 'wb') as output:
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

    # USING PANDAS IS PREFERRED
    def preProcessDataUsingPandas(self):
        debug = 0
        df = self.x  # train data
        dft = self.xTest  # test data for prediction
        columnsToEncode = list(df.select_dtypes(include=['category', 'object']))
        nonCatColumns = list(df.select_dtypes(exclude=['category', 'object']))
        if debug:
            print(nonCatColumns)
        # ONE HOT ENCODING / NUMERIC CATEGORICAL ENCODING
        for feature in columnsToEncode:
            unqNamesX = set(df[feature].unique())
            unqNamesXTest = set(dft[feature].unique())

            unq = len(df[feature].unique())
            nSamples = len(df[feature])
            if unq <= 7:
                # controlled one hot encoding, otherwise the number of columns will increase exponential
                # buggy some feature might be present in train data but not in test data or the otherway around
                # temporaray fix, just drop the column if unique values are not same for both test and train
                try:
                    if unqNamesX == unqNamesXTest:
                        if debug:
                            print("one hot:test, train-unique values SAME")
                        # for train data
                        ohCols = pd.get_dummies(df[feature], prefix=feature)
                        df = df.drop(feature, axis=1)
                        df = df.join(ohCols)

                        # for test data -
                        ohCols = pd.get_dummies(dft[feature], prefix=feature)
                        dft = dft.drop(feature, axis=1)
                        dft = dft.join(ohCols)
                    else:
                        if debug:
                            print("one hot: test, train-unique values DIFF")
                        df = df.drop(feature, axis=1)
                        dft = dft.drop(feature, axis=1)
                except:
                    print('Error one hot encoding ' + feature)
            elif nSamples > (10 * unq):
                # numerical encoding
                try:
                    if unqNamesX == unqNamesXTest:
                        if debug:
                            print("num encodeing :test, train-unique values SAME")
                        # for train data
                        df[feature] = df[feature].astype('category')
                        df[feature] = df[feature].cat.codes

                        # for test data
                        dft[feature] = dft[feature].astype('category')
                        dft[feature] = dft[feature].cat.codes
                    else:
                        if debug:
                            print("num encodeing :test, train-unique values DIFF")

                        df = df.drop(feature, axis=1)
                        dft = dft.drop(feature, axis=1)
                except:
                    print('Error numerical encoding ' + feature)
            else:
                # remove the column from the dataset
                df = df.drop(feature, axis=1)
                dft = dft.drop(feature, axis=1)

        for feature in nonCatColumns:
            nanCountX = df[feature].isnull().sum()
            nanCountXTest = dft[feature].isnull().sum()
            if debug:
                print("-------",feature)
                print(nanCountX)
                print(nanCountXTest)

            nSamples = len(df[feature])

            if nanCountX > 0.5 * nSamples:
                df = df.drop(feature, axis=1)
                dft = dft.drop(feature, axis=1)
                continue
            elif nanCountX > 0 or nanCountXTest > 0:
                if debug:
                    print("filling na values")
                # imputation handling using mean of the data in the column
                df[feature].fillna(df[feature].mean(), inplace=True)
                dft[feature].fillna(df[feature].mean(), inplace=True)
            if debug:
                print("before normalizing x featur-->", df[feature].isnull().any())
                print("before normalizing xt featur-->", dft[feature].isnull().any())
            # normalizing the data
            max_value = df[feature].max()
            min_value = df[feature].min()
            if max_value != min_value:
                df[feature] = (df[feature] - min_value) / (max_value - min_value)

                # max_value = dft[feature].max()
                # min_value = dft[feature].min()
                dft[feature] = (dft[feature] - min_value) / (max_value - min_value)
            else:
                df = df.drop(feature, axis=1)
                dft = dft.drop(feature, axis=1)
                continue
            if debug:
                print("after normalizing x featur-->", df[feature].isnull().any())
                print("after normalizing xt featur-->", dft[feature].isnull().any())
        self.x = df
        self.xTest = dft

    # ---------------------------------------------------------------------------------------------------------------------
    def catEncoderUsingSklearn(self):
        df = self.x
        columnsToEncode = list(df.select_dtypes(include=['category', 'object']))
        le = pre.LabelEncoder()
        ohe = pre.OneHotEncoder()
        for feature in columnsToEncode:
            print(feature)
            if df[feature].isnull().any():
                df[feature].fillna("NA", inplace=True)

            unq = len(np.unique(df[feature]))
            nSamples = len(df[feature])
            unqNames = np.unique(df[feature])

            if unq <= 7:
                # one hot encoding
                try:
                    temp = le.fit_transform(df[feature])
                    temp = np.split(temp, nSamples)
                    temp = list(map(lambda x: np.array(x).tolist(), temp))
                    # ohe.fit(t)
                    oh_array = ohe.fit_transform(temp).toarray()
                    for i in range(unq):
                        df[feature + "_" + unqNames[i]] = oh_array[:, i].tolist()
                    df = df.drop(feature, axis=1)
                    # we can do this using dictvectorizer also
                    # vec = fe.DictVectorizer(sparse=False)
                    # t=v.fit_transform(df[feature])
                except:
                    print('Error encoding ' + feature)
            elif nSamples > (10 * unq):
                # numerical encoding
                try:
                    df[feature] = le.fit_transform(df[feature])
                except:
                    print('Error numerical encoding ' + feature)
            else:
                # remove the column from the dataset
                df = df.drop(feature, axis=1)

        self.x = df

    # ---------------------------------------------------------------------------------------------------------------------
    # Data splitter - into train and validation sets
    def splitter(self):
        self.xTrain, self.xValidation, self.yTrain, self.yValidation = train_test_split(self.x, self.y, test_size=0.2)

        # convert them to np arrays
        self.xTrain = np.ascontiguousarray(self.xTrain)
        self.yTrain = np.ascontiguousarray(self.yTrain)
        self.xValidation = np.ascontiguousarray(self.xValidation)
        self.yValidation = np.ascontiguousarray(self.yValidation)

        self.xTest = np.ascontiguousarray(self.xTest)

    def pca(self, pcaMinVar=0.001):
        # Increase pcaMinVar for more feature reduction
        # First find the explained variance by each component
        self.pcaMinVar = pcaMinVar
        _pcaVar = PCA()
        _pcaVar.fit(self.xTrain)

        pcaNumComps = sum(_pcaVar.explained_variance_ratio_ > pcaMinVar)
        _pca = PCA(pcaNumComps)

        # Reduce the train data
        _pca.fit(self.xTrain)
        self.xTrainPca = _pca.transform(self.xTrain)
        # Reducing validation set
        _pca.fit(self.xValidation)
        self.xValidationPca = _pca.transform(self.xValidation)

        if self.xTest is not None:
            # Reducing test set
            _pca.fit(self.xTest)
            self.xTestPca = _pca.transform(self.xTest)
        else:
            print('Error: No test dataset file provided')

    def lasso(self):
        # Here first we train a linear lasso to fit the the given data set and
        # then use this model to reduce the dimensionality of the dataset.
        _lasso = Lasso(alpha=1.0, fit_intercept=True, normalize=False,
                       precompute=False, copy_X=True, max_iter=1000,
                       tol=0.0001, warm_start=False, positive=False,
                       random_state=None, selection='cyclic').fit(self.xTrain, self.yTrain)

        # Select the model that was trained.
        lassoModel = SelectFromModel(_lasso, prefit=True)
        self.xTrainLasso = lassoModel.transform(self.xTrain)
        self.xValidationLasso = lassoModel.transform(self.xValidation)

        if self.xTest is not None:
            self.xTestLasso = lassoModel.transform(self.xTest)
        else:
            print('Error: No test dataset file provided')

    def hyperOpt(self, search='random', crosVal=10, parallel=-1, num_iter=10):
        # The inputs to this method are:
        # self : Calling object
        # search: Search method - 'random' (default) or 'grid'
        # crosVal : no. of folds of cross validation (default is 10)
        # parallel : whether to run in parallel (set as -1) or not (set as 1)
        #            default is 1
        # num_iter : number of iterations to run incase of randomized search
        #            default is 10



        crosVal = np.max([crosVal, 3])

        regRidge = linear_model.Ridge()
        regLasso = linear_model.Lasso()
        regSvrRbf = SVR(kernel='rbf')
        regSvrPoly = SVR(kernel='poly')

        if search == 'grid':

            # Setting up parameter search space
            paramSvrPoly = {'C': [1, 10, 100, 1000], 'degree': [2, 3, 4, 5, 6], 'kernel': ['poly']}
            paramSvrRbf = {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf'], }
            # 'class_weight':['balanced', None]}
            paramLasso = {'alpha': [1, 0.1, 0.01, 0.001, 0.0001, 0]}
            paramRidge = {'alpha': [1, 0.1, 0.01, 0.001, 0.0001, 0]}

            # performing grid search in parameter search space (with parallelization)
            self.ModelSvrPoly = GridSearchCV(regSvrPoly, paramSvrPoly, cv=crosVal, n_jobs=parallel,
                                             verbose=1,
                                             scoring='neg_mean_absolute_error')

            self.ModelSvrRbf = GridSearchCV(regSvrRbf, paramSvrRbf, cv=crosVal, n_jobs=parallel,
                                            verbose=1,
                                            scoring='neg_mean_absolute_error')

            self.ModelLasso = GridSearchCV(regLasso, paramLasso, cv=crosVal, n_jobs=parallel,
                                           verbose=1,
                                           scoring='neg_mean_absolute_error')

            self.ModelRidge = GridSearchCV(regRidge, paramRidge, cv=crosVal, n_jobs=parallel,
                                           verbose=1,
                                           scoring='neg_mean_absolute_error')

        elif search == 'random':

            # Setting up parameter search space
            paramSvrPoly = {'C': stats.expon(scale=100), 'degree': stats.randint(2, 6), 'kernel': ['poly']}
            paramSvrRbf = {'C': stats.expon(scale=100), 'gamma': stats.expon(scale=.1), 'kernel': ['rbf'], }
            paramLasso = {'alpha': stats.lognorm(0.965)}
            paramRidge = {'alpha': stats.lognorm(0.965)}

            # performing grid search in parameter search space (with parallelization)
            self.ModelSvrPoly = RandomizedSearchCV(regSvrPoly, paramSvrPoly, cv=crosVal, n_jobs=parallel,
                                                   n_iter=num_iter,
                                                   verbose=1,
                                                   scoring='neg_mean_absolute_error')

            self.ModelSvrRbf = RandomizedSearchCV(regSvrRbf, paramSvrRbf, cv=crosVal, n_jobs=parallel, n_iter=num_iter,
                                                  scoring='neg_mean_absolute_error',
                                                  random_state=6,
                                                  verbose=1)

            self.ModelLasso = RandomizedSearchCV(regLasso, paramLasso, cv=crosVal, n_jobs=parallel, n_iter=num_iter,
                                                 scoring='neg_mean_absolute_error',
                                                 random_state=6,
                                                 verbose=1)

            self.ModelRidge = RandomizedSearchCV(regRidge, paramRidge, cv=crosVal, n_jobs=parallel, n_iter=num_iter,
                                                 scoring='neg_mean_absolute_error',
                                                 random_state=6,
                                                 verbose=1)

        # Making copy of these estimator to make different estimator
        # for PCA reduced features
        self.ModelSvrPolyPca = copy.deepcopy(self.ModelSvrPoly)
        self.ModelSvrRbfPca = copy.deepcopy(self.ModelSvrRbf)
        self.ModelRidgePca = copy.deepcopy(self.ModelRidge)
        self.ModelLassoPca = copy.deepcopy(self.ModelLasso)

        # for Lasso reduced features
        self.ModelSvrPolyLasso = copy.deepcopy(self.ModelSvrPoly)
        self.ModelSvrRbfLasso = copy.deepcopy(self.ModelSvrRbf)
        self.ModelRidgeLasso = copy.deepcopy(self.ModelRidge)
        self.ModelLassoLasso = copy.deepcopy(self.ModelLasso)

        # # Fitting the models to original train data
        self.ModelSvrPoly.fit(self.xTrain, self.yTrain)
        self.ModelSvrRbf.fit(self.xTrain, self.yTrain)
        self.ModelLasso.fit(self.xTrain, self.yTrain)
        self.ModelRidge.fit(self.xTrain, self.yTrain)

        # Fitting the models to PCA reduced train data
        # self.ModelSvrPolyPca.fit(self.xTrainPca, self.yTrain)
        # self.ModelSvrRbfPca.fit(self.xTrainPca, self.yTrain)
        # self.ModelLassoPca.fit(self.xTrainPca, self.yTrain)
        # self.ModelRidgePca.fit(self.xTrainPca, self.yTrain)

        # Fitting the models to Lasso reduced train data
        self.ModelSvrPolyLasso.fit(self.xTrainLasso, self.yTrain)
        self.ModelSvrRbfLasso.fit(self.xTrainLasso, self.yTrain)
        self.ModelLassoLasso.fit(self.xTrainLasso, self.yTrain)
        self.ModelRidgeLasso.fit(self.xTrainLasso, self.yTrain)

        # return self.self.ModelRidge

        # Extracting the best parameter values for original data
        self.CSvrPoly = self.ModelSvrPoly.best_params_['C']
        self.degreeSvrPoly = self.ModelSvrPoly.best_params_['degree']
        self.CSvrRbf = self.ModelSvrRbf.best_params_['C']
        self.gammaSvrRbf = self.ModelSvrRbf.best_params_['gamma']
        self.alphaLasso = self.ModelLasso.best_params_['alpha']
        self.alphaRidge = self.ModelRidge.best_params_['alpha']

        # Extracting the best parameter values for PCA reduced data
        # self.CSvrPolyPca = self.ModelSvrPolyPca.best_params_['C']
        # self.degreeSvrPolyPca = self.ModelSvrPolyPca.best_params_['degree']
        # self.CSvrRbfPca = self.ModelSvrRbfPca.best_params_['C']
        # self.gammaSvrRbfPca = self.ModelSvrRbfPca.best_params_['gamma']
        # self.alphaLassoPca = self.ModelLassoPca.best_params_['alpha']
        # self.alphaRidgePca = self.ModelRidgePca.best_params_['alpha']

        # Extracting the best parameter values for Lasso reduced data
        self.CSvrPolyLasso = self.ModelSvrPolyLasso.best_params_['C']
        self.degreeSvrPolyLasso = self.ModelSvrPolyLasso.best_params_['degree']
        self.CSvrRbfLasso = self.ModelSvrRbfLasso.best_params_['C']
        self.gammaSvrRbfLasso = self.ModelSvrRbfLasso.best_params_['gamma']
        self.alphaLassoLasso = self.ModelLassoLasso.best_params_['alpha']
        self.alphaRidgeLasso = self.ModelRidgeLasso.best_params_['alpha']

    def modelSelect(self):
        # This function takes calling object as input and returns the best model

        # Predicting on validation set using original data
        ySvrPoly = self.ModelSvrPoly.predict(self.xValidation)
        ySvrRbf = self.ModelSvrRbf.predict(self.xValidation)
        yLasso = self.ModelLasso.predict(self.xValidation)
        yRidge = self.ModelRidge.predict(self.xValidation)

        # Predicting on validation set using PCA reduced data
        # ySvrPolyPca = self.ModelSvrPolyPca.predict(self.xValidationPca)
        # ySvrRbfPca = self.ModelSvrRbfPca.predict(self.xValidationPca)
        # yLassoPca = self.ModelLassoPca.predict(self.xValidationPca)
        # yRidgePca = self.ModelRidgePca.predict(self.xValidationPca)

        # Predicting on validation set using Lasso reduced data
        ySvrPolyLasso = self.ModelSvrPolyLasso.predict(self.xValidationLasso)
        ySvrRbfLasso = self.ModelSvrRbfLasso.predict(self.xValidationLasso)
        yLassoLasso = self.ModelLassoLasso.predict(self.xValidationLasso)
        yRidgeLasso = self.ModelRidgeLasso.predict(self.xValidationLasso)

        # Error Evaluation on original dataset
        self.errSvrPoly = sum(abs(ySvrPoly - self.yValidation))
        self.errSvrRbf = sum(abs(ySvrRbf - self.yValidation))
        self.errLasso = sum(abs(yLasso - self.yValidation))
        self.errRidge = sum(abs(yRidge - self.yValidation))

        # Error Evaluation on PCA reduced dataset
        # self.errSvrPolyPca = sum(abs(ySvrPolyPca - self.yValidation))
        # self.errSvrRbfPca = sum(abs(ySvrRbfPca - self.yValidation))
        # self.errLassoPca = sum(abs(yLassoPca - self.yValidation))
        # self.errRidgePca = sum(abs(yRidgePca - self.yValidation))

        # Error Evaluation on lasso reduced dataset
        self.errSvrPolyLasso = sum(abs(ySvrPolyLasso - self.yValidation))
        self.errSvrRbfLasso = sum(abs(ySvrRbfLasso - self.yValidation))
        self.errLassoLasso = sum(abs(yLassoLasso - self.yValidation))
        self.errRidgeLasso = sum(abs(yRidgeLasso - self.yValidation))
        minErr = np.min([[self.errSvrRbf, self.errLasso, self.errRidge, self.errSvrPoly,
                          # self.errSvrRbfPca, self.errLassoPca, self.errRidgePca, self.errSvrPolyPca,
                          self.errSvrRbfLasso, self.errLassoLasso, self.errRidgeLasso, self.errSvrPolyLasso]])

        # Return the best model
        if minErr == self.errSvrPoly:
            self.BestModel = self.ModelSvrPoly
            self.BestModelPreproc = {'method': 'none'}
        elif minErr == self.errSvrRbf:
            self.BestModel = self.ModelSvrRbf
            self.BestModelPreproc = {'method': 'none'}
        elif minErr == self.errLasso:
            self.BestModel = self.ModelLasso
            self.BestModelPreproc = {'method': 'none'}
        elif minErr == self.errRidge:
            self.BestModel = self.ModelRidge
            self.BestModelPreproc = {'method': 'none'}
        elif minErr == self.errSvrPolyLasso:
            self.BestModel = self.ModelSvrPolyLasso
            self.BestModelPreproc = {'method': 'Lasso'}
        elif minErr == self.errSvrRbfLasso:
            self.BestModel = self.ModelSvrRbfLasso
            self.BestModelPreproc = {'method': 'Lasso'}
        elif minErr == self.errLassoLasso:
            self.BestModel = self.ModelLassoLasso
            self.BestModelPreproc = {'method': 'Lasso'}
        elif minErr == self.errRidgeLasso:
            self.BestModel = self.ModelRidgeLasso
            self.BestModelPreproc = {'method': 'Lasso'}
        # If needed remove other models from self object here

        # Save objects
        timeStr = time.strftime("%d_%b_%H:%M", time.localtime())
        self.saveObject(self.BestModel, 'BestModel' + timeStr + '.pkl')
        self.saveObject(self.ModelLasso, 'ModelLasso' + timeStr + '.pkl')
        self.saveObject(self.ModelRidge, 'ModelRidge' + timeStr + '.pkl')
        self.saveObject(self.ModelSvrRbf, 'ModelSvrRbf' + timeStr + '.pkl')
        self.saveObject(self.ModelSvrPoly, 'ModelSvrPoly' + timeStr + '.pkl')
        self.saveObject(self.ModelLassoLasso, 'ModelLassoLasso' + timeStr + '.pkl')
        self.saveObject(self.ModelRidgeLasso, 'ModelRidgeLasso' + timeStr + '.pkl')
        self.saveObject(self.ModelSvrRbfLasso, 'ModelSvrRbfLasso' + timeStr + '.pkl')
        self.saveObject(self.ModelSvrPolyLasso, 'ModelSvrPolyLasso' + timeStr + '.pkl')

        return self.BestModel



        # default function that starts executing when this file is RUN directly - use it for debugging purposes
        # if __name__ == '__main__':

    def predictForTestSet(self, yTestcsv):
        # if self.BestModelPreproc['method'] == 'PCA':
        #     yPredict = self.BestModel.predict(self.xTestPca)
        if self.BestModelPreproc['method'] == 'Lasso':
            self.yPredict = self.BestModel.predict(self.xTestLasso)
        else:
            self.yPredict = self.BestModel.predict(self.xTest)
        
        prediction = pd.DataFrame(self.yPredict, columns=[self.yLabel]).to_csv(self.yLabel+'.csv')
        
        if yTestcsv:
            yTrue = (np.genfromtxt(yTestcsv, delimiter=',', skip_header=1))
            # yTrue = np.mat(yTrue())
            self.errPerf = sum(abs(yTrue - self.yPredict)) / np.shape(yTrue)

##################################################################################################################################