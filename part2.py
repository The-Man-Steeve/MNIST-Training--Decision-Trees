from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sys import argv
def main():
    mode = argv[1]
    #Load MNIST dataset
    X, y= fetch_openml('mnist_784',version=1,return_X_y=True)
    X= X /255.0 #Normalize pixel values to [0,1]
    #Split into training (60K) and test (10K)sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=(6/7), 
        random_state=42)

    #X_train,X_test = X[:60000], X[60000:]
    #y_train,y_test = y[:60000], y[60000:]
    #---------------------------------------------------------------
    #DECISION TREE
    if mode == 'dt' or mode == 'bagging':
        dt = DecisionTreeClassifier(random_state=42)
        
        dt_grid = {
        'criterion' : ['gini', 'entropy', 'log_loss'],
        'max_depth' : [12, 16, 20],
        'max_features' : ['sqrt', 'log2'],
        }
        dt_experiment = GridSearchCV(estimator=dt, #assign the estimator
                                    param_grid=dt_grid, #assign the parameter grid
                                    n_jobs=2, #use 2 processors in parallel
                                    cv=3 #3 fold cross validation instead of the default 5
                                    )

        #dt.fit(X=X_train, y=y_train)
        dt_experiment.fit(X=X_train, y=y_train)
        dt_prediction = dt_experiment.predict(X=X_test)

        if mode == 'dt':
            output_file = open('mnist_dt.txt', mode='w')
            output_file.write("**Decision Tree Classifier**\n")
            output_file.write("Parameters: \n")
            output_file.write(str(dt_experiment.best_params_))
            output_file.write('\n')
            output_file.write("Accuracy: \n")
            output_file.write(str(accuracy_score(y_true=y_test, y_pred=dt_prediction)))
            output_file.close()
    #--------------------------------------------------------------------------------------
    #BAGGING METHOD
    if mode == 'bagging':
        #I put the best parameters from my experiment in here just to save time
        best_dt = DecisionTreeClassifier(
            criterion='entropy',
            max_depth=20,
            max_features='sqrt',
            random_state=42
        )
        #best_dt.fit(X=X_train, y=y_train)
        
        bagger = BaggingClassifier(random_state=42, estimator=best_dt)
        #bagger.fit(X=X_train, y=y_train)

        bagger_grid = {
            'n_estimators': [25, 50, 75, 100]
        }
        bagger_experiment = GridSearchCV(estimator=bagger, param_grid=bagger_grid, cv=3, n_jobs=2)
        bagger_experiment.fit(X=X_train, y=y_train)
        bagger_prediction = bagger_experiment.predict(X=X_test)
        #print("Bagging Accuracy: ", accuracy_score(y_true=y_test, y_pred=bagging_prediction))
        output_file = open("mnist_bagging.txt", 'w')
        output_file.write("**Bagging Classifier\n**")
        output_file.write("Best Parameters: \n")
        output_file.write(str(bagger_experiment.best_params_))
        output_file.write('\n')
        output_file.write("Accuracy: \n")
        output_file.write(str(accuracy_score(y_true=y_test, y_pred=bagger_prediction)))
        output_file.close()

    #-----------------------------------------------------------------------------------
    #RANDOM FOREST
    if mode == 'rf':
        rf = RandomForestClassifier(random_state=42,
                                    n_jobs=4
                                    )

        rf_grid = {
            'n_estimators': [100, 150, 200],
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': [10, 15, 20],
            'max_features': ['sqrt', 'log2'],
            'max_samples' : [.5, .75, 1]
        }

        rf_experiment = GridSearchCV(estimator=rf, param_grid=rf_grid, cv=3)

        rf_experiment.fit(X=X_train, y=y_train)
        rf_prediction = rf_experiment.predict(X=X_test)
        #print("RF Accuracy: ", accuracy_score(y_true=y_test, y_pred=rf_prediction))
        output_file = open('mnist_rf.txt', 'w')
        output_file.write("**Random Forest Classifier**\n")
        output_file.write("Best Parameters: \n")
        output_file.write(str(rf_experiment.best_params_))
        output_file.write("\n")
        output_file.write("Accuracy: \n")
        output_file.write(str(accuracy_score(y_true=y_test, y_pred=rf_prediction)))
        output_file.close()

    #-------------------------------------------------------------------------------
    #GRADIENT BOOSTING
    if mode == 'gb':
        
        gb = GradientBoostingClassifier(random_state=42)

        gb_grid = {
            'n_estimators': [100, 150, 200],
            'learning_rate': [.05, .1, .2],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [1, 3, 5]
        }

        gb_experiment = GridSearchCV(estimator=gb, param_grid=gb_grid, n_jobs=4, cv=3)
        gb_experiment.fit(X=X_train, y=y_train)
        gb_prediction = gb_experiment.predict(X_test)
        #print("GB Accuracy: ", accuracy_score(y_true=y_test, y_pred=gb_prediction))
        output_file = open('mnist_gb.txt', 'w')
        output_file.write("**Gradient Boosting Classifier**\n")
        output_file.write("Parameters: \n")
        output_file.write(str(gb_experiment.best_params_))
        output_file.write('\n')
        output_file.write("Accuracy: \n")
        output_file.write(str(accuracy_score(y_true=y_test, y_pred=gb_prediction)))
        output_file.close()

if __name__ == '__main__':
    main()