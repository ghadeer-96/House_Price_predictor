#!/usr/bin/env python
# coding: utf-8

# author : Ghadeer Abualrub


import pandas as pd
import numpy as np
import re
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression,ElasticNet,Ridge,SGDRegressor
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, KFold,RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler,PolynomialFeatures




class FeatureEngineering:
    @staticmethod
    def rearrange_features(df):
        df = df.iloc[:, ~df.columns.str.contains('^Unnamed')]
        df.update(df[df['house_price'].str.contains('K')].iloc[:,0].str.replace('K','000'))
        df = df.drop(columns=['am'])
        df=df.astype('int')
        return df

    @staticmethod
    def na_percentage_in_rows(df):
        is_NaN = df. isnull()
        row_has_NaN = is_NaN. any(axis=1)
        rows_with_NaN = df[row_has_NaN]
        #df[row_has_NaN].index

        stat = pd.DataFrame()
        stat['row'] = df[row_has_NaN].index
        stat['na percentage'] = ((36-df[row_has_NaN].apply(lambda x: x.count(), axis=1))/36).tolist()
        return stat,stat[stat['na percentage']>=0.5].iloc[:,0]
    
    @staticmethod
    def na_percentage_in_cols(df):
        stat =pd.DataFrame()
        stat['col'] = df.columns
        stat['na percentage']=df.isna().mean().tolist()
        return stat,stat[stat['na percentage']>=0.5].iloc[:,0]

    @staticmethod
    def handle_missings(df):
        s1,index=FeatureEngineering.na_percentage_in_rows(df)
        s2,features=FeatureEngineering.na_percentage_in_cols(df)
        print(features)
        df = df.drop(index)
        if features.size!=0:
            df = df.drop(columns = [features])
        return df


    # Outlier Treatment
    # We can drop the outliers as we have sufficient data.
    # outlier treatment for price
    def drop_outliers(df,data_series):
        Q1, Q3 = df[data_series].quantile([0.25, 0.75]).values
        IQR = Q3 - Q1
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR

        s=df[data_series][( df[data_series] < lower_limit) |
                          ( df[data_series] > upper_limit) ]
        s.index
        df = df.drop(s.index)
        return df

        #return data_series[( data_series >= lower_limit) &
        #                  ( data_series <= upper_limit) ]


class FeatureSelection:
    @staticmethod
    def select_features(df, target_variable):
        
        x = df.drop(target_variable, 1)
        y = df[target_variable]
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=37)
        
        model = rfc(n_estimators = 300, n_jobs = -1,random_state =37, min_samples_leaf = 50)
        
        sfm = SelectFromModel(model,threshold=0.02)
        sfm.fit(x_train, y_train)
        selected_features = x_train.columns[(sfm.get_support())]
        
        # Creating a bar plot
        font = {'size'   : 7}
        matplotlib.rc('font', **font)
        model.fit(x_train, y_train)
        feature_imp = pd.Series(model.feature_importances_,index=x.columns.values).sort_values(ascending=False)
        sns.barplot(x=feature_imp, y=feature_imp.index)
        # Add labels to your graph
        plt.xlabel('Feature Importance Score')
        plt.ylabel('Features')
        plt.title("Visualizing Important Features")
        plt.legend()
        plt.show()

        return selected_features
    





class dataSplitter:
    def __init__(self,df,target_variable,selected_feat):
        #self.x = df.drop(target_variable, 1)
        self.x = df[selected_feat]
        self.y = df[target_variable]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x,self.y,test_size=0.2,random_state=37)
    
    def scale_features(self):
        pipeline = Pipeline([
            ('std_scalar', StandardScaler())
        ])

        self.x_train = pipeline.fit_transform(self.x_train)
        self.x_test = pipeline.transform(self.x_test)





class Model:

    def cross_val(self,model,x,y):
        cv = KFold(n_splits=10, random_state=1, shuffle=True)
        # evaluate model
        pred = cross_val_score(model, x, y,cv=cv)
        return pred.mean()



    def print_evaluate(self,true, predicted):  
        mae = mean_absolute_error(true, predicted)
        mse = mean_squared_error(true, predicted)
        rmse = np.sqrt(mean_squared_error(true, predicted))
        r2_square = r2_score(true, predicted)
        print('MAE:', mae)
        print('MSE:', mse)
        print('RMSE:', rmse)
        print('R2 Square', r2_square)
        print('__________________________________')


    def evaluate(self,true, predicted):
        mae = mean_absolute_error(true, predicted)
        mse = mean_squared_error(true, predicted)
        rmse = np.sqrt(mean_squared_error(true, predicted))
        r2_square = r2_score(true, predicted)
        return mae, mse, rmse, r2_square   
    
    
    def SaveModel(self,filename='finalized_model.sav'):
        # Save the model as a pickle in a file
        # save the model to disk
        pickle.dump(model, open(filename, 'wb'))
        # some time later...
        # load the model from disk
    #load
        #loaded_model = pickle.load(open(filename, 'rb'))



class ModelTraining(Model):
    def __init__(self, data, dataTune):
        
        self.results_df = pd.DataFrame()
        self.data = data
        self.dataTune = dataTune
        
    def linearRegTrain(self):
        # define object of linear regression model
        model = LinearRegression(normalize=True)
        # train the model
        model.fit(self.data.x_train,self.data.y_train)
        
        test_pred = model.predict(self.data.x_test)
        train_pred = model.predict(self.data.x_train)

        print('Test dataset evaluation:\n_____________________________________')
        print_evaluate(self.data.y_test, test_pred)
        print('Train dataset evaluation:\n_____________________________________')
        print_evaluate(self.data.y_train, train_pred)
        self.results_df = pd.DataFrame(data=[["Linear Regression", *self.evaluate(self.data.y_test, test_pred) , self.cross_val(model,self.data.x_test,self.data.y_test)]], 
                          columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])
        self.SaveModel('LinReg.sav')
        return model
    

##################### Random Forest Regressor#########################   
    def RandomForestTrain(self):

        model = self.HyperTuneRandomForest()
        model.fit(self.data.x_train, self.data.y_train)

        test_pred = model.predict(self.data.x_test)
        train_pred = model.predict(self.data.x_train)

        print('Test set evaluation:\n_____________________________________')
        print_evaluate(self.data.y_test, test_pred)
        print('Train set evaluation:\n_____________________________________')
        print_evaluate(self.data.y_train, train_pred)
        results_df_1 = pd.DataFrame(data=[["Random Forest Regressor", *self.evaluate(self.data.y_test, test_pred), self.cross_val(model,self.data.x_test,self.data.y_test)]], 
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Cross Validation'])
        self.results_df = self.results_df.append(results_df_1, ignore_index=True)
        self.SaveModel('RandForest.sav')
        return model
    
    def HyperTuneRandomForest(self):
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        
        rf = RandomForestRegressor()
        # Random search of parameters, using 3 fold cross validation, 
        # search across 100 different combinations, and use all available cores
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
        # Fit the random search model
        rf_random.fit(self.dataTune.x_train, self.dataTune.y_train)
        return rf_random.best_estimator_
    
###################### Gradient Boosting Regressor ####################

    def GradientBoostingTrain(self):
          
        model = self.HyperTuneGradientBoosting()
        model.fit(self.data.x_train, self.data.y_train)
        y_pred = model.predict(self.data.x_test)
        
        print('Test dataset evaluation:\n_____________________________________')
        print_evaluate(self.data.y_test, test_pred)
        print('Train dataset evaluation:\n_____________________________________')
        print_evaluate(self.data.y_train, train_pred)
        results_df_1 = pd.DataFrame(data=[["Gradient Boosting Regressor", *self.evaluate(self.data.y_test, test_pred), self.cross_val(model,self.data.x_test,self.data.y_test)]], 
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Cross Validation'])
        self.results_df = self.results_df.append(results_df_1, ignore_index=True)
        self.SaveModel('GradBoost.sav')
        return model
    
    def HyperTuneGradientBoosting(self):
        params = {'n_estimators':[500, 1000, 1500, 2000], 'max_depth':[3, 5, 8],'random_state':[22,37,50]}
        gbr = GradientBoostingRegressor()
        gbr_grid = GridSearchCV(gbr, params, cv=5)
        gbr_grid.fit(self.dataTune.x_train, self.dataTune.y_train)
        return gbr_grid.best_estimator_
    
#################### ElasticNet ####################

    def ElasticNetTrain(self):
        model = ElasticNet(alpha=0.1, l1_ratio=0.9, selection='random', random_state=42)
        model.fit(self.data.x_train, self.data.y_train)

        test_pred = model.predict(self.data.x_test)
        train_pred = model.predict(self.data.x_train)

        print('Test set evaluation:\n_____________________________________')
        print_evaluate(self.data.y_test, test_pred)
        print('====================================')
        print('Train set evaluation:\n_____________________________________')
        print_evaluate(self.data.y_train, train_pred)
        results_df_1 = pd.DataFrame(data=[["ElasticNet Regressor", *self.evaluate(self.data.y_test, test_pred), self.cross_val(model,self.data.x_test,self.data.y_test)]], 
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Cross Validation'])
        self.results_df = self.results_df.append(results_df_1, ignore_index=True)
        self.SaveModel('Elastic.sav')
        return model

##################### Lasso ##################
    def LassoTrain(self): 
        model = Lasso(alpha=0.1, 
                      precompute=True, 
                      positive=True, 
                      selection='random',
                      random_state=42)
        model.fit(self.data.x_train, self.data.y_train)

        test_pred = model.predict(self.data.x_test)
        train_pred = model.predict(self.data.x_train)

        print('Test set evaluation:\n_____________________________________')
        print_evaluate(self.data.y_test, test_pred)
        print('====================================')
        print('Train set evaluation:\n_____________________________________')
        print_evaluate(self.data.y_train, train_pred)
        results_df_1 = pd.DataFrame(data=[["Lasso Regressor", *self.evaluate(self.data.y_test, test_pred), self.cross_val(model,self.data.x_test,self.data.y_test)]], 
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Cross Validation'])
        self.results_df = self.results_df.append(results_df_1, ignore_index=True)
        self.SaveModel('Lasso.sav')
        return model




