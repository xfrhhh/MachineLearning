import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from mlxtend.regressor import StackingCVRegressor
import lightgbm as lgb
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import warnings


def ignore_warn(*args, **kwargs):
    pass


warnings.warn = ignore_warn

train = pd.read_csv('train_data.csv')
test = pd.read_csv('test_data.csv')
print('The shape of training data:', train.shape)
print('The shape of testing data:', test.shape)
# 查看目标值的斜度和峰度

y = train['SalePrice']
print('Skewness of target:', y.skew())
print('kurtosis of target:', y.kurtosis())
# sns.histplot(y, kde=True)
# 明显右偏，取对数
y = np.log1p(y)
print('Skewness of target:', y.skew())
print('kurtosis of target:', y.kurtosis())
sns.histplot(y, kde=True)
plt.show()

train = train.drop('SalePrice', axis=1)

# 检查训练集与测试集的维度是否一致
print('The shape of training data:', train.shape)
print('The length of y:', len(y))
print('The shape of testing data:', test.shape)

# 采用十折交叉验证
n_folds = 10


def rmse_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=20)
    rmse = np.sqrt(-cross_val_score(model, train.values, y,
                                    scoring='neg_mean_squared_error', cv=kf))
    return (rmse)


# Lasso
lasso_alpha = [0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
lasso = make_pipeline(RobustScaler(), LassoCV(alphas=lasso_alpha, random_state=2))

# LightGBM
lgbr_params = {'learning_rate': 0.01,
               'n_estimators': 1850,
               'max_depth': 4,
               'num_leaves': 20,
               'subsample': 0.6,
               'colsample_bytree': 0.6,
               'min_child_weight': 0.001,
               'min_child_samples': 21,
               'random_state': 42,
               'reg_alpha': 0,
               'reg_lambda': 0.05}
lgbr = lgb.LGBMRegressor(**lgbr_params)

# 用之前设定的评估方法进行评估
models_name = ['Lasso', 'LightGBM']
models = [lasso, lgbr]
for i, model in enumerate(models):
    score = rmse_cv(model)
    print('{} score: {}({})'.format(models_name[i], score.mean(), score.std()))

# 模型融合,构建Stacking模型
stack_model = StackingCVRegressor(regressors=(lasso, lgbr), meta_regressor=lasso,
                                  use_features_in_secondary=True)

# 整个训练集上训练各个模型
# Lasso
lasso_trained = lasso.fit(np.array(train), np.array(y))

# LightGBM
lgbr_trained = lgbr.fit(np.array(train), np.array(y))

# Stacking
stack_model_trained = stack_model.fit(np.array(train), np.array(y))


# 评估各个模型在训练集上的表现
# 先定义评估方法，采用kaggle规定的评估方法
def rmse(y, y_preds):
    return np.sqrt(mean_squared_error(y, y_preds))


# 评估模型
models.append(stack_model)
models_name.append('Stacking_model')
for i, model in enumerate(models):
    y_preds = model.predict(np.array(train))
    model_score = rmse(y, y_preds)
    print('RMSE of {}: {}'.format(models_name[i], model_score))

# 提交预测结果
sample_submission = pd.read_csv('sample_submission.csv')
for i, model in enumerate(models):
    preds = model.predict(np.array(test))
    submission = pd.DataFrame({'Id': sample_submission['Id'], 'SalePrice': np.expm1(preds)})
    submission.to_csv('house_prices_submission_' + models_name[i] + '_optimation.csv', index=False)
    print('{} finished.'.format(models_name[i]))

# 均值融合
preds_in_train = np.zeros((len(y), len(models)))
for i, model in enumerate(models):
    preds_in_train[:, i] = model.predict(np.array(train))
average_preds_in_train = preds_in_train.mean(axis=1)
average_score = rmse(y, average_preds_in_train)
print('RMSE of average model on training data:', average_score)

# 提交均值融合预测结果
preds_in_test = np.zeros((len(test), len(models)))
for i, model in enumerate(models):
    preds_in_test[:, i] = model.predict(np.array(test))
average_preds_in_test = preds_in_test.mean(axis=1)

average_score = rmse(y, average_preds_in_train)
print('RMSE of average model on training data:', average_score)

average_submission = pd.DataFrame({'Id': sample_submission['Id'], 'SalePrice': np.expm1(average_preds_in_test)})
average_submission.to_csv('House_Price_submission_average_model_optimation.csv', index=False)
