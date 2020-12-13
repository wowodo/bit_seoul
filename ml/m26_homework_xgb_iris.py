#파이프 라인까지 구성할것
#
from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

parameters = [
    {"xgb__n_estimators": [100, 200, 300], "xgb__learning_rate": [0.1, 0.3, 0.001, 0.01], "xgb__max_depth":[4, 5, 6]},
    {"xgb__n_estimators": [90, 100, 110], "xgb__learning_rate": [0.1, 0.001, 0.01], "xgb__max_depth":[4, 5, 6], "xgb__colsample_bytree": [0.6, 0.9, 1]},
    {"xgb__n_estimators": [90, 110], "xgb__learning_rate": [0.1, 0.001, 0.5], "xgb__max_depth":[4, 5, 6], "xgb__colsample_bytree":[0.6, 0.9, 1],
    "xgb__colsample": [0.6, 0.7, 0.9]}
]

#score 롬 성능비교

from sklearn.model_selection import RandomizedSearchCV, KFold,cross_val_score

datasets = load_iris()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    datasets.data, datasets.target, random_state=66, shuffle=True, train_size=0.8
)

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

pipe = Pipeline([("scaler", MinMaxScaler()), ("xgb", XGBClassifier())])

kfold = KFold(n_splits=5, shuffle=True)
model = RandomizedSearchCV(pipe, parameters, cv=kfold, verbose=2)

model.fit(x_train, y_train)
print("최적의 매개변수 :",model.best_estimator_)

from sklearn.metrics import accuracy_score
score = model.score(x_test, y_test) #회귀는   r2_score = score 이다

print("model.score", score)

'''
최적의 매개변수 : Pipeline(steps=[('scaler', MinMaxScaler()),
                ('xgb',
                 XGBClassifier(base_score=0.5, booster='gbtree',
                               colsample_bylevel=1, colsample_bynode=1,
                               colsample_bytree=1, gamma=0, gpu_id=-1,
                               importance_type='gain',
                               interaction_constraints='', learning_rate=0.3,
                               max_delta_step=0, max_depth=6,
                               min_child_weight=1, missing=nan,
                               monotone_constraints='()', n_estimators=200,
                               n_jobs=0, num_parallel_tree=1,
                               objective='multi:softprob', random_state=0,
                               reg_alpha=0, reg_lambda=1, scale_pos_weight=None,
                               subsample=1, tree_method='exact',
                               validate_parameters=1, verbosity=None))])
model.score 0.9
'''


