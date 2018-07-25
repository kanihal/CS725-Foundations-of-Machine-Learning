from autoML import AutoRegressor
import time
csvFile = './datasets/housing_train_SalePrice.csv'
csvFileTest = './datasets/housing_test_SalePrice.csv'
yLabel = "SalePrice"

ar = AutoRegressor(trainCsvFile=csvFile, testCsvFile=csvFileTest, yLabel=yLabel)
ar.preProcessDataUsingPandas()
ar.splitter()
ar.lasso()

start_time = time.time()
ar.hyperOpt()
print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
ar.modelSelect()
print("--- %s seconds ---" % (time.time() - start_time))
ar.predictForTestSet()
yp=ar.yPredict

# To read pickle file
# with open('BestModel.pkl', 'rb') as f:
# data = pickle.load(f)