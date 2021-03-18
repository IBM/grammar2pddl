from lale.lib.sklearn import *
from lale.lib.lale import NoOp

def get_baseline_planned_pipelines() :
    ret = [{
        'lale_pipeline' : 
        (
            NoOp | Normalizer | MinMaxScaler | RobustScaler |
            StandardScaler | PCA
        ) >> (
            LogisticRegression | DecisionTreeClassifier |
            RandomForestClassifier | GradientBoostingClassifier |
            ExtraTreesClassifier | GaussianNB | KNeighborsClassifier |
            QuadraticDiscriminantAnalysis
        ),
        'pipeline' : '(NoOp | Nrm | MMS | RS | SS | PCA) >> (LR | DT | RF | GB | ET | GNB | KN | QDA)'
    }, {
        'lale_pipeline' :
        (
            NoOp | Normalizer | MinMaxScaler | RobustScaler | StandardScaler
        ) >> (
            NoOp | PCA
        ) >> (
            LogisticRegression | DecisionTreeClassifier |
            RandomForestClassifier | GradientBoostingClassifier |
            ExtraTreesClassifier | GaussianNB | KNeighborsClassifier |
            QuadraticDiscriminantAnalysis
        ),
        'pipeline' : '(NoOp | Nrm | MMS | RS | SS) >> (NoOp | PCA) >> (LR | DT | RF | GB | ET | GNB | KN | QDA)'
    }]

    return ret
# --
