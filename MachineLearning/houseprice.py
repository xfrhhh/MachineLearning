import matplotlib
import numpy as np

import pandas as pd
from jedi.api.refactoring import inline

pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x))

import seaborn as sns

color = sns.color_palette()
sns.set_style('darkgrid')

import matplotlib.pyplot as plt
# %matplotlib inline

from scipy import stats
from scipy.special import boxcox1p
from scipy.stats import norm, skew

# #忽略警告
# import warnings
# def ignore_warn(*args, **kwargs):
#     pass
# warnings.warn = ignore_warn

from sklearn.preprocessing import LabelEncoder

# 解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 查看训练集数据 The shape of training data: (1460, 81) id:1 ~ 1460
train = pd.read_csv('train.csv')
print('The shape of training data:', train.shape)
print(train.head())
# 查看测试集数据 The shape of testing data: (1459, 80) id:1461 ~ 2919
test = pd.read_csv('test.csv')
print('The shape of testing data:', test.shape)
print(test.head())
# 删除训练集和测试集的 Id 列 The shape of training data: (1460, 80);The shape of testing data: (1459, 79)
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)
print('The shape of training data:', train.shape)
print('The shape of testing data:', test.shape)

# 数据分析
# 绘制目标值分布
sns.histplot(train['SalePrice'])

# 目标值的"统计值：
print('目标值的统计值:')
print(train['SalePrice'].describe())

# 分离数字特征和类别特征
num_features = []
cate_features = []
for col in test.columns:
    if test[col].dtype == 'object':
        cate_features.append(col)
    else:
        num_features.append(col)
print('数字特征:', len(num_features))  # 36
print('类别特征:', len(cate_features))  # 43

# 查看数字特征与目标值的关系
# plt.figure()
# plt.subplots_adjust(hspace=2, wspace=1)
# for i, feature in enumerate(num_features):
#     plt.subplot(9, 4, i + 1)
#     sns.scatterplot(x=feature, y='SalePrice', data=train, alpha=0.5)
#     plt.xlabel(feature)
#     plt.ylabel('SalePrice')
# plt.show()

# 查看类别特征与目标值的关系
# 查看‘Neighborhood’与目标值的关系
plt.figure()
sns.boxplot(y='Neighborhood', x='SalePrice', data=train, color='skyblue')
# add stripplot
# ax = sns.stripplot(x='Neighborhood', y='SalePrice', data=train, color="blue", jitter=0.2, size=4)

plt.xlabel('SalePrice', fontsize=14)
plt.ylabel('Neighborhood', fontsize=14)
plt.xticks(rotation=0, fontsize=12)
# plt.show()

# 出售年份’YrSold’和房价'SalePrice'的关系 //查看结果关系不大
plt.figure(figsize=(16, 10))
sns.boxplot(x='YrSold', y='SalePrice', data=train)
plt.xlabel('YrSold', fontsize=14)
plt.ylabel('SalePrice', fontsize=14)
plt.xticks(rotation=90, fontsize=12)
plt.title("出售年份’YrSold’和房价'SalePrice'的关系")
# plt.show()

# 所有特征之间的相关关系
corrs = train.corr()
plt.figure(figsize=(16, 16))
sns.heatmap(corrs)
# plt.show()

# 分析与目标值相关度最高的十个变量
plt.figure(figsize=(6, 6))
cols_10 = corrs.nlargest(10, 'SalePrice')['SalePrice'].index
corrs_10 = train[cols_10].corr()

# 设置展示一半，如果不需要注释掉mask即可
mask = np.zeros_like(corrs_10)
mask[np.triu_indices_from(mask)] = True  # np.triu_indices 上三角矩阵


sns.heatmap(corrs_10, annot=True, cmap='YlGnBu', mask=mask)
plt.xticks(rotation=30)
plt.show()

# 绘制这十个特征两两之间的散点图
g = sns.PairGrid(train[cols_10])
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
# plt.show()

# ‘TotalBsmtSF’、‘GrLiveArea’、'Neighborhood’这几个是我们要重点关注的特征。

# 异常值处理
plt.figure()
sns.scatterplot(x='TotalBsmtSF', y='SalePrice', data=train)
# plt.show() //无法显示
# 处理掉右下的明显异常值

train = train.drop(train[(train['TotalBsmtSF'] > 6000) & (train['SalePrice'] < 200000)].index)

sns.scatterplot(x='TotalBsmtSF', y='SalePrice', data=train)
# plt.show()

# 对’GrLiveArea’进行同样的处理
plt.figure()
# sns.scatterplot(x='GrLivArea', y='SalePrice', data=train)
# 处理掉右下的异常值
train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 200000)].index)

# sns.scatterplot(x='GrLivArea', y='SalePrice', data=train)
# plt.show()

# 查看训练集中各特征的数据缺失个数
print('The shape of training data:', train.shape)
train_missing = train.isnull().sum()
train_missing = train_missing.drop(train_missing[train_missing == 0].index).sort_values(ascending=False)
print("训练集中各特征的数据缺失个数:")
print(train_missing)

# 查看测试集中各特征的数据缺失个数
print('The shape of testing data:', test.shape)
test_missing = test.isnull().sum()
test_missing = test_missing.drop(test_missing[test_missing == 0].index).sort_values(ascending=False)
print("测试集中各特征的数据缺失个数:")
print(test_missing)

# 根据特征说明文档，以下特征缺失代表没有，所以直接补充为’None’
none_lists = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType',
              'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtFinType1',
              'BsmtFinType2', 'BsmtCond', 'BsmtExposure', 'BsmtQual', 'MasVnrType']
for col in none_lists:
    train[col] = train[col].fillna('None')
    test[col] = test[col].fillna('None')

# 根据文档，以下特征缺失不代表没有，只是说明数据丢失了，所以对缺失值补充该特征中出现次数最多的值
most_lists = ['MSZoning', 'Exterior1st', 'Exterior2nd', 'SaleType', 'KitchenQual', 'Electrical']
for col in most_lists:
    train[col] = train[col].fillna(train[col].mode()[0])
    test[col] = test[col].fillna(train[col].mode()[0])

    # 注意这里补充的是训练集中出现最多的类别

# ‘Functional’这个特征，文档中说缺失值都看作是’Typ’，所以直接填入’Typ’
train['Functional'] = train['Functional'].fillna('Typ')
test['Functional'] = test['Functional'].fillna('Typ')
train.drop('Utilities', axis=1, inplace=True)
test.drop('Utilities', axis=1, inplace=True)

# 参照特征说明文档，对可能为零的特征，缺失值全部补零
zero_lists = ['GarageYrBlt', 'MasVnrArea', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
              'GarageCars', 'GarageArea',
              'TotalBsmtSF']
for col in zero_lists:
    train[col] = train[col].fillna(0)
    test[col] = test[col].fillna(0)

# 对不能为零的特征，按’Neighborhood’分组，补充为同类’Neighborhood’中该特征的中位数
train['LotFrontage'] = train.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.median()))
for ind in test['LotFrontage'][test['LotFrontage'].isnull().values == True].index:
    x = test['Neighborhood'].iloc[ind]
    test['LotFrontage'].iloc[:][ind] = train.groupby('Neighborhood')['LotFrontage'].median()[x]

# 检查是否还存在缺失值
print("检查是否还存在缺失值:")
print(train.isnull().sum().any())
print(test.isnull().sum().any())

# 转换类别特征
# 从存放类别特征的列表去掉'Utilities'
cate_features.remove('Utilities')
print('The number of categorical features:', len(cate_features))

for col in cate_features:
    train[col] = train[col].astype(str)
    test[col] = test[col].astype(str)
le_features = ['Street', 'Alley', 'LotShape', 'LandContour', 'LandSlope', 'HouseStyle', 'RoofMatl', 'Exterior1st',
               'Exterior2nd', 'ExterQual',
               'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
               'HeatingQC', 'CentralAir',
               'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive',
               'PoolQC', 'Fence']
for col in le_features:
    encoder = LabelEncoder()
    value_train = set(train[col].unique())
    value_test = set(test[col].unique())
    value_list = list(value_train | value_test)
    encoder.fit(value_list)
    train[col] = encoder.transform(train[col])
    test[col] = encoder.transform(test[col])

# 处理偏斜特征
# 先获得偏斜度大于0.5的特征，我们要对这一部分进行处理
skewness = train[num_features].apply(lambda x: skew(x)).sort_values(ascending=False)
skewness = skewness[skewness > 0.5]
skew_features = skewness.index
print(skewness)
# 由于数据中可能存在许多没有处理的异常值，为了增强模型对异常值的刚度，我们采用Box Cox转换来处理偏斜数据：
for col in skew_features:
    lam = stats.boxcox_normmax(train[col] + 1)  # +1是为了保证输入大于零
    train[col] = boxcox1p(train[col], lam)
    test[col] = boxcox1p(test[col], lam)

# 围绕与目标值相关度大的几个特征来构造新的特征
train['IsRemod'] = 1
train['IsRemod'].loc[train['YearBuilt'] == train['YearRemodAdd']] = 0  # 是否翻新(翻新：1， 未翻新：0)
train['BltRemodDiff'] = train['YearRemodAdd'] - train['YearBuilt']  # 翻新与建造的时间差（年）
train['BsmtUnfRatio'] = 0
train['BsmtUnfRatio'].loc[train['TotalBsmtSF'] != 0] = train['BsmtUnfSF'] / train['TotalBsmtSF']  # Basement未完成占总面积的比例
train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']  # 总面积
# 对测试集做同样的处理
test['IsRemod'] = 1
test['IsRemod'].loc[test['YearBuilt'] == test['YearRemodAdd']] = 0  # 是否翻新(翻新：1， 未翻新：0)
test['BltRemodDiff'] = test['YearRemodAdd'] - test['YearBuilt']  # 翻新与建造的时间差（年）
test['BsmtUnfRatio'] = 0
test['BsmtUnfRatio'].loc[test['TotalBsmtSF'] != 0] = test['BsmtUnfSF'] / test['TotalBsmtSF']  # Basement未完成占总面积的比例
test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']  # 总面积

dummy_features = list(set(cate_features).difference(set(le_features)))
print(dummy_features)

all_data = pd.concat((train.drop('SalePrice', axis=1), test)).reset_index(drop=True)
all_data = pd.get_dummies(all_data, drop_first=True)  # 注意独热编码生成的时候要去掉一个维度，保证剩下的变量都是相互独立的

trainset = all_data[:1458]
y = train['SalePrice']
trainset['SalePrice'] = y.values
testset = all_data[1458:]
print('The shape of training data:', trainset.shape)
print('The shape of testing data:', testset.shape)
# 保存处理后的数据
trainset.to_csv('train_data.csv', index=False)
testset.to_csv('test_data.csv', index=False)

# plt.show()
