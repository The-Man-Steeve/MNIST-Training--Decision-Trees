from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import itertools
from copy import deepcopy
import sys

def main():
    #GATHER USER INPUT
    c_index = sys.argv[1]
    d_index = sys.argv[2]
    mode = sys.argv[3]
    print("**EXPERIMENT**")
    print("C = ", c_index)
    print("D = ", d_index)
    print("Mode = ", mode)
    train_file = 'all_data\\train_c' + c_index + '_d' + d_index + '.csv'
    test_file = 'all_data\\test_c' + c_index + '_d' + d_index + '.csv'
    validation_file = 'all_data\\valid_c' + c_index + '_d' + d_index + '.csv'

    '''
    train_file = 'all_data\\train_c500_d100.csv'
    test_file = 'all_data\\test_c500_d100.csv'
    validation_file = 'all_data\\valid_c500_d100.csv'
    '''
    #EXTRACT THE DATA FROM THE FILES
    #Initially code inspired by AI
    train = pd.read_csv(train_file, header=None)
    x_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]

    test = pd.read_csv(test_file, header=None)
    x_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1]

    valid = pd.read_csv(validation_file, header=None)
    x_valid = valid.iloc[:, :-1]
    y_valid = valid.iloc[:, -1]

    combined_tv_x = pd.concat([x_train,x_valid])
    combined_tv_y = pd.concat([y_train, y_valid])
    #DECISION TREE-------------------------------------------------------------------------
    if mode == 'dt' or mode == 'bagging':
        #SET PARAMETERS FOR TUNING
        parameter_grid = {
            'criterion' : ['gini', 'entropy', 'log_loss'],
            'max_depth' : [6, 7, 8, 9, 10, 11, 12, 13],
            'min_samples_split' : [2, .01, .05, .1],
            'min_samples_leaf' : [1, 5, 10, .01, .05, .1],
            'max_features' : [None, 'sqrt', 'log2']
        }
        #this creates a way to iterate through every combination of parameters
        #I created this with the help of AI
        combos = list(itertools.product(
            parameter_grid['criterion'],
            parameter_grid['max_depth'],
            parameter_grid['max_features'],
            parameter_grid['min_samples_leaf'],
            parameter_grid['min_samples_split'],
        ))

        best_parameters = None
        best_accuracy = 0
        #print('Testing Hyperparameters...')
        for c, md, mf, msl, mss in combos:
            exp = DecisionTreeClassifier(
                criterion=c,
                max_depth=md,
                max_features=mf,
                min_samples_leaf=msl,
                min_samples_split=mss,
            )
            exp.fit(x_train,y_train)
            validation_prediction = exp.predict(x_valid)
            validation_accuracy = accuracy_score(y_true=y_valid, y_pred=validation_prediction)
            if validation_accuracy > best_accuracy:
                #print('new best parameter combination found!')
                best_accuracy = validation_accuracy
                best_parameters = {
                    'criterion': c,
                    'max_depth': md,
                    'max_features': mf,
                    'min_samples_leaf': msl,
                    'min_samples_split': mss,
                }
        if mode == 'dt':
            print("Best Hyperparameters: ")
            for element in best_parameters:
                print(element, ': ', best_parameters[element])
        #print('Validation Accuracy: ', best_accuracy)
        #print('Validation F1 Score: ', best_f1)

        final_dt = DecisionTreeClassifier(
            criterion=best_parameters['criterion'],
            max_depth=best_parameters['max_depth'],
            max_features=best_parameters['max_features'],
            min_samples_leaf=best_parameters['min_samples_leaf'],
            min_samples_split=best_parameters['min_samples_split'],
        )
        #print("Training final model...")
        final_dt.fit(X=combined_tv_x, y=combined_tv_y)
        test_prediction = final_dt.predict(x_test)
        if mode == 'dt':
            print("Final Results on Test Data:")
            dt_accuracy = accuracy_score(y_true=y_test, y_pred=test_prediction)
            dt_f1 = f1_score(y_true=y_test, y_pred=test_prediction)
            print("Accuracy: ", dt_accuracy)
            print('F1 Score: ', dt_f1)

        #BAGGING------------------------------------------------------------------------
        if mode == 'bagging':
            best_bagging_accuracy = 0
            best_bagging_parameters = None
            bagging_parameters = {
                'num_estimators': [10, 20, 30, 40, 50],
                'max_samples': [.05, .1, .15, .2, .25]
            }
            b_combos = list(itertools.product(
                bagging_parameters['num_estimators'],
                bagging_parameters['max_samples']
            ))
            #iterate through combinations
            for n, m in b_combos:
                bagger = BaggingClassifier(
                    estimator=final_dt,
                    n_estimators=n,
                    max_samples=m
                    )
                bagger.fit(x_train, y_train)
                b_prediction = bagger.predict(x_valid)
                b_prediction_acc = accuracy_score(y_true=y_valid, y_pred=b_prediction)
                if b_prediction_acc > best_bagging_accuracy:
                    best_bagging_accuracy = b_prediction_acc
                    best_bagging_parameters = {
                        'n_estimators' : n,
                        'max_samples' : m
                    }
            
            print("Best Parameters:")
            for p in best_bagging_parameters:
                print(p, ': ', best_bagging_parameters[p])
            #retrain best model
            bagger = BaggingClassifier(
                n_estimators= best_bagging_parameters['n_estimators'],
                max_samples= best_bagging_parameters['max_samples']
            )
            bagger.fit(X=combined_tv_x, y=combined_tv_y)
            #print("Training final model: ")


            final_b_prediction = bagger.predict(X=x_test)
            print("Accuracy: ", accuracy_score(y_true=y_test, y_pred=final_b_prediction))
            print("F1 Score: ", f1_score(y_true=y_test, y_pred=final_b_prediction))
    #RANDOM FOREST-----------------------------------------------------------------------------
    if mode == 'rf':
        print("Constructing the random forest...")
        best_rf_params = None
        best_rf_accuracy = 0
        rf_params = {
            'criterion' : ['gini', 'entropy', 'log_loss'],
            'n_estimators': [50, 100, 150],
            'max_depth': [5, 10, 15],
            'min_samples_split': [.01, .05, .1],
            'min_samples_leaf': [.01, .05, .1],
            'max_features': ['sqrt', 'log2'],
            'max_samples' : [.5, .75, .9]
        }
        rf_combos = list(itertools.product(
            rf_params['criterion'],
            rf_params['n_estimators'],
            rf_params['max_depth'],
            rf_params['min_samples_split'],
            rf_params['min_samples_leaf'],
            rf_params['max_features'],
            rf_params['max_samples']
        ))
        for c, ne, md, mss, msl, mf, ms in rf_combos:
            rf = RandomForestClassifier(
                n_estimators=ne,
                criterion=c,
                max_depth=md,
                min_samples_split=mss,
                min_samples_leaf=msl,
                max_features=mf,
                max_samples=ms,
                n_jobs=4
            )
            rf.fit(X=x_train, y=y_train)
            rf_validation = rf.predict(X=x_valid)
            current_accuracy = accuracy_score(y_true=y_valid, y_pred=rf_validation)
            if current_accuracy > best_rf_accuracy:
                best_rf_accuracy = current_accuracy
                best_rf_params = {
                    'n_estimators': ne,
                    'criterion' : c,
                    'max_depth' : md,
                    'min_samples_split' : mss,
                    'min_samples_leaf' : msl,
                    'max_features' : mf,
                    'max_samples' : ms
                }
        for p in best_rf_params:
            print(p, " : ", best_rf_params[p])
        rf = RandomForestClassifier(
            n_estimators= best_rf_params['n_estimators'],
            criterion= best_rf_params['criterion'],
            max_depth= best_rf_params['max_depth'],
            min_samples_split= best_rf_params['min_samples_split'],
            min_samples_leaf= best_rf_params['min_samples_leaf'],
            max_features= best_rf_params['max_features'],
            max_samples= best_rf_params['max_samples']
        )
        #print("**Testing on Test Set**")
        rf.fit(X=combined_tv_x, y=combined_tv_y)
        final_rf_prediction = rf.predict(X=x_test)
        print("Accuracy: ", accuracy_score(y_true=y_test, y_pred=final_rf_prediction))
        print('F1 Score: ', f1_score(y_true=y_test, y_pred=final_rf_prediction))

    #GRADIENT BOOSTING:
    if mode == 'gb':
        print("Gradient Boosting...")
        best_gb_classifier = None
        best_gb_accuracy = 0
        gb_f1 = 0
        gb_params = {
            'loss' : ['log_loss', 'exponential'],
            'learning_rate': [.01, .05, .1, .25],
            'n_estimators': [50, 100, 150],
            'criterion' : ['friedman_mse', 'squared_error'],
            'max_depth' : [1, 3],
            'max_features' : ['sqrt', 'log2', None],
        }
        gb_combos = list(itertools.product(
            gb_params['criterion'],
            gb_params['learning_rate'],
            gb_params['loss'],
            gb_params['max_depth'],
            gb_params['max_features'],
            gb_params['n_estimators']
        ))
        for c, lr, l, md, mf, ne in gb_combos:
            gb = GradientBoostingClassifier(
                criterion=c,
                learning_rate=lr,
                loss=l,
                max_depth=md,
                max_features=mf,
                n_estimators=ne
            )
            gb.fit(X=x_train, y=y_train)
            gb_prediction = gb.predict(X=x_valid)
            current_accuracy = accuracy_score(y_true=y_valid, y_pred=gb_prediction)
            if current_accuracy > best_gb_accuracy:
                best_gb_accuracy = current_accuracy
                gb_f1 = f1_score(y_true=y_valid, y_pred=gb_prediction)
                best_gb_classifier = deepcopy(gb)
        print("Best Parameters for Gradient Boosting:")
        print(best_gb_classifier.get_params())
        print("**Testing on Test Set**")
        gb.fit(X=combined_tv_x, y=combined_tv_y)
        final_gb_prediction = gb.predict(X=x_test)
        print("Accuracy: ", accuracy_score(y_true=y_test, y_pred=final_gb_prediction))
        print('F1 Score: ', f1_score(y_true=y_test, y_pred=final_gb_prediction))

if __name__ == '__main__':
    main()