# -------------------- Python Functions ------------------------ #
# Version: V1.0
# Author: Kaggler
# -------------------------------------------------------------- #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 1. 缺失率统计 -----------------------------------------------------------------------------------------


def find_missing(data):
    # number of missing values
    count_missing = data.isnull().sum().values
    # total records
    total = data.shape[0]
    # percentage of missing
    ratio_missing = count_missing / total
    # return a dataframe to show: feature name, # of missing and % of missing
    return pd.DataFrame(data={'missing_count': count_missing, 'missing_ratio': ratio_missing}, index=data.columns.values)

# find_missing(application_train).head(12)
# ---------------------------------------------------------------------------------------------------

# 2. 分布图  -----------------------------------------------------------------------------------------
# 2.1 Categorical ----------------


def plot_categorical(data, col, size=[8, 4], xlabel_angle=0, title=''):
    '''use this for ploting the count of categorical features'''
    plotdata = data[col].value_counts()
    plt.figure(figsize=size)
    sns.barplot(x=plotdata.index, y=plotdata.values)
    plt.title(title)
    if xlabel_angle != 0:
        plt.xticks(rotation=xlabel_angle)   # 轴坐标刻度-角度
    plt.show()
# plot_categorical(data=application_train, col='TARGET', size=[8 ,4], xlabel_angle=0, title='train set: label')

# 2.2 Numerical -------------------


def plot_numerical(data, col, size=[8, 4], bins=50):
    '''use this for ploting the distribution of numercial features'''
    plt.figure(figsize=size)
    plt.title("Distribution of %s" % col)
    sns.distplot(data[col].dropna(), kde=True, bins=bins)  # 排除缺失值
    plt.show()
# plot_numerical(application_train, 'AMT_CREDIT')

# 2.3 Categorical features by label -----------------


def plot_categorical_bylabel(data, col, by_col, size=[12, 6], xlabel_angle=0, title=''):
    '''use it to compare the distribution between label 1 and label 0'''
    plt.figure(figsize=size)
    l1 = data.loc[data[by_col] == 1, col].value_counts()
    l0 = data.loc[data[by_col] == 0, col].value_counts()
    plt.subplot(1, 2, 1)
    sns.barplot(x=l1.index, y=l1.values)
    plt.title('Default: ' + title)
    plt.xticks(rotation=xlabel_angle)
    plt.subplot(1, 2, 2)
    sns.barplot(x=l0.index, y=l0.values)
    plt.title('Non-default: ' + title)
    plt.xticks(rotation=xlabel_angle)
    plt.show()


# plot_categorical_bylabel(application_train, 'CODE_GENDER', title='Gender')

# 2.4 Numerical features by label -----------------


def plot_numerical_bylabel(data, col, by_col, size=[8, 4], bins=50):
    '''use this to compare the distribution of numercial features'''
    plt.figure(figsize=[12, 6])
    l1 = data.loc[data[by_col] == 1, col]
    l0 = data.loc[data[by_col] == 0, col]
    plt.subplot(1, 2, 1)
    sns.distplot(l1.dropna(), kde=True, bins=bins)
    plt.title('Default: Distribution of %s' % col)
    plt.subplot(1, 2, 2)
    sns.distplot(l0.dropna(), kde=True, bins=bins)
    plt.title('Non-default: Distribution of %s' % col)
    plt.show()

# plot_numerical_bylabel(application_train, 'EXT_SOURCE_1', bins=50)
# ---------------------------------------------------------------------------------------------------
