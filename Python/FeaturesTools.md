
# Introduction: Automated Feature Engineering

In this notebook, we will look at an exciting development in data science: automated feature engineering. A machine learning model can only learn from the data we give it, and making sure that data is relevant to the task is one of the most crucial steps in the machine learning pipeline (this is made clear in the excellent paper ["A Few Useful Things to Know about Machine Learning"](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf)). 

However, manual feature engineering is a tedious task and is limited by both human imagination - there are only so many features we can think to create - and by time - creating new features is time-intensive. Ideally, there would be an objective method to create an array of diverse new candidate features that we can then use for a machine learning task. This process is meant to not replace the data scientist, but to make her job easier and allowing her to supplement domain knowledge with an automated workflow.

In this notebook, we will walk through an implementation of using [Featuretools](https://www.featuretools.com/), an open-source Python library for automatically creating features with relational data (where the data is in structured tables). Although there are now many efforts working to enable automated model selection and hyperparameter tuning, there has been a lack of automating work on the feature engineering aspect of the pipeline. This library seeks to close that gap and the general methodology has been proven effective in both [machine learning competitions with the data science machine](https://github.com/HDI-Project/Data-Science-Machine) and [business use cases](https://www.featurelabs.com/blog/predicting-credit-card-fraud/). 


## Dataset

To show the basic idea of featuretools we will use an example dataset consisting of three tables:

* `clients`: information about clients at a credit union
* `loans`: previous loans taken out by the clients
* `payments`: payments made/missed on the previous loans

The general problem of feature engineering is taking disparate data, often distributed across multiple tables, and combining it into a single table that can be used for training a machine learning model. Featuretools has the ability to do this for us, creating many new candidate features with minimal effort. These features are combined into a single table that can then be passed on to our model. 

First, let's load in the data and look at the problem we are working with.


```python
# Run this if featuretools is not already installed
# !pip install -U featuretools
```


```python
# pandas and numpy for data manipulation
import pandas as pd
import numpy as np

# featuretools for automated feature engineering
import featuretools as ft

# ignore warnings from pandas
import warnings
warnings.filterwarnings('ignore')
```


```python
# Read in the data
clients = pd.read_csv('/Users/chenys/Desktop/clients.csv', parse_dates = ['joined'])
loans = pd.read_csv('/Users/chenys/Desktop/loan.csv', parse_dates = ['loan_start', 'loan_end'])
payments = pd.read_csv('/Users/chenys/Desktop/payments.csv', parse_dates = ['payment_date'])
```


```python
clients.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>client_id</th>
      <th>joined</th>
      <th>income</th>
      <th>credit_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>46109</td>
      <td>2002-04-16</td>
      <td>172677</td>
      <td>527</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49545</td>
      <td>2007-11-14</td>
      <td>104564</td>
      <td>770</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41480</td>
      <td>2013-03-11</td>
      <td>122607</td>
      <td>585</td>
    </tr>
    <tr>
      <th>3</th>
      <td>46180</td>
      <td>2001-11-06</td>
      <td>43851</td>
      <td>562</td>
    </tr>
    <tr>
      <th>4</th>
      <td>25707</td>
      <td>2006-10-06</td>
      <td>211422</td>
      <td>621</td>
    </tr>
  </tbody>
</table>
</div>




```python
loans.sample(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>client_id</th>
      <th>loan_type</th>
      <th>loan_amount</th>
      <th>repaid</th>
      <th>loan_id</th>
      <th>loan_start</th>
      <th>loan_end</th>
      <th>rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>428</th>
      <td>26945</td>
      <td>credit</td>
      <td>6010</td>
      <td>0</td>
      <td>10078</td>
      <td>2005-03-06</td>
      <td>2006-11-29</td>
      <td>2.64</td>
    </tr>
    <tr>
      <th>75</th>
      <td>46180</td>
      <td>cash</td>
      <td>8320</td>
      <td>0</td>
      <td>10670</td>
      <td>2005-08-24</td>
      <td>2008-05-16</td>
      <td>5.51</td>
    </tr>
    <tr>
      <th>129</th>
      <td>32726</td>
      <td>cash</td>
      <td>10671</td>
      <td>1</td>
      <td>11520</td>
      <td>2002-03-02</td>
      <td>2004-09-01</td>
      <td>3.74</td>
    </tr>
    <tr>
      <th>135</th>
      <td>32726</td>
      <td>home</td>
      <td>3754</td>
      <td>1</td>
      <td>10049</td>
      <td>2012-01-27</td>
      <td>2013-09-15</td>
      <td>3.31</td>
    </tr>
    <tr>
      <th>83</th>
      <td>25707</td>
      <td>home</td>
      <td>2203</td>
      <td>0</td>
      <td>10363</td>
      <td>2014-02-11</td>
      <td>2015-08-20</td>
      <td>7.40</td>
    </tr>
    <tr>
      <th>273</th>
      <td>44601</td>
      <td>credit</td>
      <td>7232</td>
      <td>1</td>
      <td>10475</td>
      <td>2005-08-19</td>
      <td>2007-12-07</td>
      <td>3.59</td>
    </tr>
    <tr>
      <th>95</th>
      <td>25707</td>
      <td>credit</td>
      <td>10636</td>
      <td>0</td>
      <td>10562</td>
      <td>2000-12-29</td>
      <td>2003-05-24</td>
      <td>4.59</td>
    </tr>
    <tr>
      <th>1</th>
      <td>46109</td>
      <td>credit</td>
      <td>9794</td>
      <td>0</td>
      <td>10984</td>
      <td>2003-10-21</td>
      <td>2005-07-17</td>
      <td>1.25</td>
    </tr>
    <tr>
      <th>337</th>
      <td>39384</td>
      <td>cash</td>
      <td>3557</td>
      <td>0</td>
      <td>11100</td>
      <td>2014-04-21</td>
      <td>2016-12-29</td>
      <td>4.16</td>
    </tr>
    <tr>
      <th>67</th>
      <td>46180</td>
      <td>cash</td>
      <td>6985</td>
      <td>1</td>
      <td>11555</td>
      <td>2004-02-04</td>
      <td>2005-10-10</td>
      <td>0.57</td>
    </tr>
  </tbody>
</table>
</div>




```python
payments.sample(10)
payments.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_id</th>
      <th>payment_amount</th>
      <th>payment_date</th>
      <th>missed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10243</td>
      <td>2369</td>
      <td>2002-05-31</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10243</td>
      <td>2439</td>
      <td>2002-06-18</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10243</td>
      <td>2662</td>
      <td>2002-06-29</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10243</td>
      <td>2268</td>
      <td>2002-07-20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10243</td>
      <td>2027</td>
      <td>2002-07-31</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>10243</td>
      <td>2243</td>
      <td>2002-09-16</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>10984</td>
      <td>1466</td>
      <td>2003-12-29</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>10984</td>
      <td>1887</td>
      <td>2004-02-01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>10984</td>
      <td>1360</td>
      <td>2004-03-09</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10984</td>
      <td>1350</td>
      <td>2004-03-29</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Manual Feature Engineering Examples

Let's show a few examples of features we might make by hand. We will keep this relatively simple to avoid doing too much work! First we will focus on a single dataframe before combining them together. In the `clients` dataframe, we can take the month of the `joined` column and the natural log of the `income` column. Later, we see these are known in featuretools as transformation feature primitives because they act on column in a single table. 


```python
# Create a month column
clients['join_month'] = clients['joined'].dt.month

# Create a log of income column
clients['log_income'] = np.log(clients['income'])

clients.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>client_id</th>
      <th>joined</th>
      <th>income</th>
      <th>credit_score</th>
      <th>join_month</th>
      <th>log_income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>46109</td>
      <td>2002-04-16</td>
      <td>172677</td>
      <td>527</td>
      <td>4</td>
      <td>12.059178</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49545</td>
      <td>2007-11-14</td>
      <td>104564</td>
      <td>770</td>
      <td>11</td>
      <td>11.557555</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41480</td>
      <td>2013-03-11</td>
      <td>122607</td>
      <td>585</td>
      <td>3</td>
      <td>11.716739</td>
    </tr>
    <tr>
      <th>3</th>
      <td>46180</td>
      <td>2001-11-06</td>
      <td>43851</td>
      <td>562</td>
      <td>11</td>
      <td>10.688553</td>
    </tr>
    <tr>
      <th>4</th>
      <td>25707</td>
      <td>2006-10-06</td>
      <td>211422</td>
      <td>621</td>
      <td>10</td>
      <td>12.261611</td>
    </tr>
  </tbody>
</table>
</div>



To incorporate information about the other tables, we use the `df.groupby` method, followed by a suitable aggregation function, followed by `df.merge`.  For example, let's calculate the average, minimum, and maximum amount of previous loans for each client. In the terms of featuretools, this would be considered an aggregation feature primitive because we using multiple tables in a one-to-many relationship to calculate aggregation figures (don't worry, this will be explained shortly!).


```python
# Groupby client id and calculate mean, max, min previous loan size
stats = loans.groupby('client_id')['loan_amount'].agg(['mean', 'max', 'min'])
stats.columns = ['mean_loan_amount', 'max_loan_amount', 'min_loan_amount']
stats.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean_loan_amount</th>
      <th>max_loan_amount</th>
      <th>min_loan_amount</th>
    </tr>
    <tr>
      <th>client_id</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25707</th>
      <td>7963.950000</td>
      <td>13913</td>
      <td>1212</td>
    </tr>
    <tr>
      <th>26326</th>
      <td>7270.062500</td>
      <td>13464</td>
      <td>1164</td>
    </tr>
    <tr>
      <th>26695</th>
      <td>7824.722222</td>
      <td>14865</td>
      <td>2389</td>
    </tr>
    <tr>
      <th>26945</th>
      <td>7125.933333</td>
      <td>14593</td>
      <td>653</td>
    </tr>
    <tr>
      <th>29841</th>
      <td>9813.000000</td>
      <td>14837</td>
      <td>2778</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Merge with the clients dataframe
clients.merge(stats, left_on = 'client_id', right_index=True, how = 'left').head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>client_id</th>
      <th>joined</th>
      <th>income</th>
      <th>credit_score</th>
      <th>join_month</th>
      <th>log_income</th>
      <th>mean_loan_amount</th>
      <th>max_loan_amount</th>
      <th>min_loan_amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>46109</td>
      <td>2002-04-16</td>
      <td>172677</td>
      <td>527</td>
      <td>4</td>
      <td>12.059178</td>
      <td>8951.600000</td>
      <td>14049</td>
      <td>559</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49545</td>
      <td>2007-11-14</td>
      <td>104564</td>
      <td>770</td>
      <td>11</td>
      <td>11.557555</td>
      <td>10289.300000</td>
      <td>14971</td>
      <td>3851</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41480</td>
      <td>2013-03-11</td>
      <td>122607</td>
      <td>585</td>
      <td>3</td>
      <td>11.716739</td>
      <td>7894.850000</td>
      <td>14399</td>
      <td>811</td>
    </tr>
    <tr>
      <th>3</th>
      <td>46180</td>
      <td>2001-11-06</td>
      <td>43851</td>
      <td>562</td>
      <td>11</td>
      <td>10.688553</td>
      <td>7700.850000</td>
      <td>14081</td>
      <td>1607</td>
    </tr>
    <tr>
      <th>4</th>
      <td>25707</td>
      <td>2006-10-06</td>
      <td>211422</td>
      <td>621</td>
      <td>10</td>
      <td>12.261611</td>
      <td>7963.950000</td>
      <td>13913</td>
      <td>1212</td>
    </tr>
    <tr>
      <th>5</th>
      <td>39505</td>
      <td>2011-10-14</td>
      <td>153873</td>
      <td>610</td>
      <td>10</td>
      <td>11.943883</td>
      <td>7424.050000</td>
      <td>14575</td>
      <td>904</td>
    </tr>
    <tr>
      <th>6</th>
      <td>32726</td>
      <td>2006-05-01</td>
      <td>235705</td>
      <td>730</td>
      <td>5</td>
      <td>12.370336</td>
      <td>6633.263158</td>
      <td>14802</td>
      <td>851</td>
    </tr>
    <tr>
      <th>7</th>
      <td>35089</td>
      <td>2010-03-01</td>
      <td>131176</td>
      <td>771</td>
      <td>3</td>
      <td>11.784295</td>
      <td>6939.200000</td>
      <td>13194</td>
      <td>773</td>
    </tr>
    <tr>
      <th>8</th>
      <td>35214</td>
      <td>2003-08-08</td>
      <td>95849</td>
      <td>696</td>
      <td>8</td>
      <td>11.470529</td>
      <td>7173.555556</td>
      <td>14767</td>
      <td>667</td>
    </tr>
    <tr>
      <th>9</th>
      <td>48177</td>
      <td>2008-06-09</td>
      <td>190632</td>
      <td>769</td>
      <td>6</td>
      <td>12.158100</td>
      <td>7424.368421</td>
      <td>14740</td>
      <td>659</td>
    </tr>
  </tbody>
</table>
</div>



We could go further and include information about `payments` in the `clients` dataframe. To do so, we would have to group `payments` by the `loan_id`, merge it with the `loans`, group the resulting dataframe by the `client_id`, and then merge it into the `clients` dataframe. This would allow us to include information about previous payments for each client. 

Clearly, this process of manual feature engineering can grow quite tedious with many columns and multiple tables and I certainly don't want to have to do this process by hand! Luckily, featuretools can automatically perform this entire process and will create more features than we would have ever thought of. Although I love `pandas`, there is only so much manual data manipulation I'm willing to stand! 

# Featuretools

Now that we know what we are trying to avoid (tedious manual feature engineering), let's figure out how to automate this process. Featuretools operates on an idea known as [Deep Feature Synthesis](https://docs.featuretools.com/api_reference.html#deep-feature-synthesis). You can read the [original paper here](http://www.jmaxkanter.com/static/papers/DSAA_DSM_2015.pdf), and although it's quite readable, it's not necessary to understand the details to do automated feature engineering. The concept of Deep Feature Synthesis is to use basic building blocks known as feature primitives (like the transformations and aggregations done above) that can be stacked on top of each other to form new features. The depth of a "deep feature" is equal to the number of stacked primitives. 

I threw out some terms there, but don't worry because we'll cover them as we go. Featuretools builds on simple ideas to create a powerful method, and we will build up our understanding in much the same way. 

The first part of Featuretools to understand [is an `entity`](https://docs.featuretools.com/loading_data/using_entitysets.html#adding-entities). This is simply a table, or in `pandas`, a `DataFrame`. We corral multiple entities into a [single object called an `EntitySet`](https://docs.featuretools.com/loading_data/using_entitysets.html). This is just a large data structure composed of many individual entities and the relationships between them.  

## EntitySet

Creating a new `EntitySet` is pretty simple: 


```python
es = ft.EntitySet(id = 'clients')
```


```python
es
```




    Entityset: clients
      Entities:
      Relationships:
        No relationships



## Entities 

An entity is simply a table, which is represented in Pandas as a `dataframe`. Each entity must have a uniquely identifying column, known as an index. For the clients dataframe, this is the `client_id` because each id only appears once in the `clients` data. In the `loans` dataframe, `client_id` is not an index because each id might appear more than once. The index for this dataframe is instead `loan_id`. 

When we create an `entity` in featuretools, we have to identify which column of the dataframe is the index. If the data does not have a unique index we can tell featuretools to make an index for the entity by passing in `make_index = True` and specifying a name for the index. If the data also has a uniquely identifying time index, we can pass that in as the `time_index` parameter. 

Featuretools will automatically infer the variable types (numeric, categorical, datetime) of the columns in our data, but we can also pass in specific datatypes to override this behavior. As an example, even though the `repaid` column in the `loans` dataframe is represented as an integer, we can tell featuretools that this is a categorical feature since it can only take on two discrete values. This is done using an integer with the variables as keys and the feature types as values.

In the code below we create the three entities and add them to the `EntitySet`.  The syntax is relatively straightforward with a few notes: for the `payments` dataframe we need to make an index, for the `loans` dataframe, we specify that `repaid` is a categorical variable, and for the `payments` dataframe, we specify that `missed` is a categorical feature. 


```python
clients
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>client_id</th>
      <th>joined</th>
      <th>income</th>
      <th>credit_score</th>
      <th>join_month</th>
      <th>log_income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>46109</td>
      <td>2002-04-16</td>
      <td>172677</td>
      <td>527</td>
      <td>4</td>
      <td>12.059178</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49545</td>
      <td>2007-11-14</td>
      <td>104564</td>
      <td>770</td>
      <td>11</td>
      <td>11.557555</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41480</td>
      <td>2013-03-11</td>
      <td>122607</td>
      <td>585</td>
      <td>3</td>
      <td>11.716739</td>
    </tr>
    <tr>
      <th>3</th>
      <td>46180</td>
      <td>2001-11-06</td>
      <td>43851</td>
      <td>562</td>
      <td>11</td>
      <td>10.688553</td>
    </tr>
    <tr>
      <th>4</th>
      <td>25707</td>
      <td>2006-10-06</td>
      <td>211422</td>
      <td>621</td>
      <td>10</td>
      <td>12.261611</td>
    </tr>
    <tr>
      <th>5</th>
      <td>39505</td>
      <td>2011-10-14</td>
      <td>153873</td>
      <td>610</td>
      <td>10</td>
      <td>11.943883</td>
    </tr>
    <tr>
      <th>6</th>
      <td>32726</td>
      <td>2006-05-01</td>
      <td>235705</td>
      <td>730</td>
      <td>5</td>
      <td>12.370336</td>
    </tr>
    <tr>
      <th>7</th>
      <td>35089</td>
      <td>2010-03-01</td>
      <td>131176</td>
      <td>771</td>
      <td>3</td>
      <td>11.784295</td>
    </tr>
    <tr>
      <th>8</th>
      <td>35214</td>
      <td>2003-08-08</td>
      <td>95849</td>
      <td>696</td>
      <td>8</td>
      <td>11.470529</td>
    </tr>
    <tr>
      <th>9</th>
      <td>48177</td>
      <td>2008-06-09</td>
      <td>190632</td>
      <td>769</td>
      <td>6</td>
      <td>12.158100</td>
    </tr>
    <tr>
      <th>10</th>
      <td>26326</td>
      <td>2004-05-06</td>
      <td>227920</td>
      <td>633</td>
      <td>5</td>
      <td>12.336750</td>
    </tr>
    <tr>
      <th>11</th>
      <td>42320</td>
      <td>2000-04-27</td>
      <td>229481</td>
      <td>563</td>
      <td>4</td>
      <td>12.343576</td>
    </tr>
    <tr>
      <th>12</th>
      <td>32961</td>
      <td>2009-04-07</td>
      <td>230341</td>
      <td>714</td>
      <td>4</td>
      <td>12.347316</td>
    </tr>
    <tr>
      <th>13</th>
      <td>29841</td>
      <td>2002-08-17</td>
      <td>38354</td>
      <td>523</td>
      <td>8</td>
      <td>10.554614</td>
    </tr>
    <tr>
      <th>14</th>
      <td>44601</td>
      <td>2004-10-20</td>
      <td>156341</td>
      <td>518</td>
      <td>10</td>
      <td>11.959795</td>
    </tr>
    <tr>
      <th>15</th>
      <td>32885</td>
      <td>2002-05-13</td>
      <td>58955</td>
      <td>642</td>
      <td>5</td>
      <td>10.984530</td>
    </tr>
    <tr>
      <th>16</th>
      <td>49068</td>
      <td>2004-02-12</td>
      <td>128813</td>
      <td>603</td>
      <td>2</td>
      <td>11.766117</td>
    </tr>
    <tr>
      <th>17</th>
      <td>44387</td>
      <td>2009-07-14</td>
      <td>151903</td>
      <td>781</td>
      <td>7</td>
      <td>11.930997</td>
    </tr>
    <tr>
      <th>18</th>
      <td>39384</td>
      <td>2000-06-18</td>
      <td>191204</td>
      <td>617</td>
      <td>6</td>
      <td>12.161096</td>
    </tr>
    <tr>
      <th>19</th>
      <td>26695</td>
      <td>2004-08-27</td>
      <td>174532</td>
      <td>680</td>
      <td>8</td>
      <td>12.069863</td>
    </tr>
    <tr>
      <th>20</th>
      <td>38537</td>
      <td>2002-10-21</td>
      <td>127183</td>
      <td>643</td>
      <td>10</td>
      <td>11.753382</td>
    </tr>
    <tr>
      <th>21</th>
      <td>46958</td>
      <td>2011-07-22</td>
      <td>225709</td>
      <td>644</td>
      <td>7</td>
      <td>12.327002</td>
    </tr>
    <tr>
      <th>22</th>
      <td>41472</td>
      <td>2001-11-06</td>
      <td>152214</td>
      <td>638</td>
      <td>11</td>
      <td>11.933043</td>
    </tr>
    <tr>
      <th>23</th>
      <td>49624</td>
      <td>2012-08-04</td>
      <td>49036</td>
      <td>800</td>
      <td>8</td>
      <td>10.800310</td>
    </tr>
    <tr>
      <th>24</th>
      <td>26945</td>
      <td>2000-11-26</td>
      <td>214516</td>
      <td>806</td>
      <td>11</td>
      <td>12.276140</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create an entity from the client dataframe
# This dataframe already has an index and a time index
es = es.entity_from_dataframe(entity_id = 'clients', dataframe = clients, 
                              index = 'client_id', time_index = 'joined')
```


```python
es
```




    Entityset: clients
      Entities:
        clients [Rows: 25, Columns: 6]
      Relationships:
        No relationships




```python
loans.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>client_id</th>
      <th>loan_type</th>
      <th>loan_amount</th>
      <th>repaid</th>
      <th>loan_id</th>
      <th>loan_start</th>
      <th>loan_end</th>
      <th>rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>46109</td>
      <td>home</td>
      <td>13672</td>
      <td>0</td>
      <td>10243</td>
      <td>2002-04-16</td>
      <td>2003-12-20</td>
      <td>2.15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>46109</td>
      <td>credit</td>
      <td>9794</td>
      <td>0</td>
      <td>10984</td>
      <td>2003-10-21</td>
      <td>2005-07-17</td>
      <td>1.25</td>
    </tr>
    <tr>
      <th>2</th>
      <td>46109</td>
      <td>home</td>
      <td>12734</td>
      <td>1</td>
      <td>10990</td>
      <td>2006-02-01</td>
      <td>2007-07-05</td>
      <td>0.68</td>
    </tr>
    <tr>
      <th>3</th>
      <td>46109</td>
      <td>cash</td>
      <td>12518</td>
      <td>1</td>
      <td>10596</td>
      <td>2010-12-08</td>
      <td>2013-05-05</td>
      <td>1.24</td>
    </tr>
    <tr>
      <th>4</th>
      <td>46109</td>
      <td>credit</td>
      <td>14049</td>
      <td>1</td>
      <td>11415</td>
      <td>2010-07-07</td>
      <td>2012-05-21</td>
      <td>3.13</td>
    </tr>
    <tr>
      <th>5</th>
      <td>46109</td>
      <td>home</td>
      <td>6935</td>
      <td>0</td>
      <td>11501</td>
      <td>2006-09-17</td>
      <td>2008-11-26</td>
      <td>1.94</td>
    </tr>
    <tr>
      <th>6</th>
      <td>46109</td>
      <td>cash</td>
      <td>6177</td>
      <td>1</td>
      <td>11141</td>
      <td>2007-03-12</td>
      <td>2009-04-26</td>
      <td>9.48</td>
    </tr>
    <tr>
      <th>7</th>
      <td>46109</td>
      <td>home</td>
      <td>12656</td>
      <td>0</td>
      <td>11658</td>
      <td>2006-05-26</td>
      <td>2007-10-15</td>
      <td>4.14</td>
    </tr>
    <tr>
      <th>8</th>
      <td>46109</td>
      <td>home</td>
      <td>11062</td>
      <td>1</td>
      <td>11611</td>
      <td>2012-09-12</td>
      <td>2014-03-14</td>
      <td>5.48</td>
    </tr>
    <tr>
      <th>9</th>
      <td>46109</td>
      <td>other</td>
      <td>4050</td>
      <td>1</td>
      <td>10828</td>
      <td>2003-12-06</td>
      <td>2005-08-19</td>
      <td>4.26</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create an entity from the loans dataframe
# This dataframe already has an index and a time index
es = es.entity_from_dataframe(entity_id = 'loans', 
                              dataframe = loans, 
                              variable_types = {'repaid': ft.variable_types.Categorical},
                              index = 'loan_id', 
                              time_index = 'loan_start')
```


```python
es
```




    Entityset: clients
      Entities:
        clients [Rows: 25, Columns: 6]
        loans [Rows: 443, Columns: 8]
      Relationships:
        No relationships




```python
# Create an entity from the payments dataframe
# This does not yet have a unique index
es = es.entity_from_dataframe(entity_id = 'payments', 
                              dataframe = payments,
                              variable_types = {'missed': ft.variable_types.Categorical},
                              make_index = True,
                              index = 'payment_id',
                              time_index = 'payment_date')
```


```python
es
```




    Entityset: clients
      Entities:
        clients [Rows: 25, Columns: 6]
        loans [Rows: 443, Columns: 8]
        payments [Rows: 3456, Columns: 5]
      Relationships:
        No relationships



All three entities have been successfully added to the `EntitySet`. We can access any of the entities using Python dictionary syntax.


```python
es['loans']
```




    Entity: loans
      Variables:
        client_id (dtype: numeric)
        loan_type (dtype: categorical)
        loan_amount (dtype: numeric)
        loan_start (dtype: datetime_time_index)
        loan_end (dtype: datetime)
        rate (dtype: numeric)
        repaid (dtype: categorical)
        loan_id (dtype: index)
      Shape:
        (Rows: 443, Columns: 8)



Featuretools correctly inferred each of the datatypes when we made this entity. We can also see that we overrode the type for the `repaid` feature, changing if from numeric to categorical. 


```python
es['payments']
```




    Entity: payments
      Variables:
        loan_id (dtype: numeric)
        payment_amount (dtype: numeric)
        payment_date (dtype: datetime_time_index)
        missed (dtype: categorical)
        payment_id (dtype: index)
      Shape:
        (Rows: 3456, Columns: 5)



## Relationships

After defining the entities (tables) in an `EntitySet`, we now need to tell featuretools [how they are related with a relationship](https://docs.featuretools.com/loading_data/using_entitysets.html#adding-a-relationship). The most intuitive way to think of relationships is with the parent to child analogy: a parent-to-child relationship is one-to-many because for each parent, there can be multiple children. The `client` dataframe is therefore the parent of the `loans` dataframe because while there is only one row for each client in the `client` dataframe, each client may have several previous loans covering multiple rows in the `loans` dataframe. Likewise, the `loans` dataframe is the parent of the `payments` dataframe because each loan will have multiple payments. 

These relationships are what allow us to group together datapoints using aggregation primitives and then create new features. As an example, we can group all of the previous loans associated with one client and find the average loan amount. We will discuss the features themselves more in a little bit, but for now let's define the relationships. 

To define relationships, we need to specify the parent variable and the child variable. This is the variable that links two entities together. In our example, the `client` and `loans` dataframes are linked together by the `client_id` column. Again, this is a parent to child relationship because for each `client_id` in the parent `client` dataframe, there may be multiple entries of the same `client_id` in the child `loans` dataframe. 

We codify relationships in the language of featuretools by specifying the parent variable and then the child variable. After creating a relationship, we add it to the `EntitySet`. 


```python
# Relationship between clients and previous loans
r_client_previous = ft.Relationship(es['clients']['client_id'],
                                    es['loans']['client_id'])

# Add the relationship to the entity set
es = es.add_relationship(r_client_previous)
```


```python
es
```




    Entityset: clients
      Entities:
        clients [Rows: 25, Columns: 6]
        loans [Rows: 443, Columns: 8]
        payments [Rows: 3456, Columns: 5]
      Relationships:
        loans.client_id -> clients.client_id



The relationship has now been stored in the entity set. The second relationship is between the `loans` and `payments`. These two entities are related by the `loan_id` variable.


```python
# Relationship between previous loans and previous payments
r_payments = ft.Relationship(es['loans']['loan_id'],
                                      es['payments']['loan_id'])

# Add the relationship to the entity set
es = es.add_relationship(r_payments)

es
```




    Entityset: clients
      Entities:
        clients [Rows: 25, Columns: 6]
        loans [Rows: 443, Columns: 8]
        payments [Rows: 3456, Columns: 5]
      Relationships:
        loans.client_id -> clients.client_id
        payments.loan_id -> loans.loan_id



We now have our entities in an entityset along with the relationships between them. We can now start to making new features from all of the tables using stacks of feature primitives to form deep features. First, let's cover feature primitives.


## Feature Primitives

A [feature primitive](https://docs.featuretools.com/automated_feature_engineering/primitives.html) a at a very high-level is an operation applied to data to create a feature. These represent very simple calculations that can be stacked on top of each other to create complex features. Feature primitives fall into two categories:

* __Aggregation__: function that groups together child datapoints for each parent and then calculates a statistic such as mean, min, max, or standard deviation. An example is calculating the maximum loan amount for each client. An aggregation works across multiple tables using relationships between tables.
* __Transformation__: an operation applied to one or more columns in a single table. An example would be extracting the day from dates, or finding the difference between two columns in one table.

Let's take a look at feature primitives in featuretools. We can view the list of primitives:


```python
primitives = ft.list_primitives()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>type</th>
      <th>description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>trend</td>
      <td>aggregation</td>
      <td>Calculates the slope of the linear trend of va...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>max</td>
      <td>aggregation</td>
      <td>Finds the maximum non-null value of a numeric ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mode</td>
      <td>aggregation</td>
      <td>Finds the most common element in a categorical...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>skew</td>
      <td>aggregation</td>
      <td>Computes the skewness of a data set.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>min</td>
      <td>aggregation</td>
      <td>Finds the minimum non-null value of a numeric ...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>sum</td>
      <td>aggregation</td>
      <td>Counts the number of elements of a numeric or ...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>time_since_last</td>
      <td>aggregation</td>
      <td>Time since last related instance.</td>
    </tr>
    <tr>
      <th>7</th>
      <td>count</td>
      <td>aggregation</td>
      <td>Counts the number of non null values.</td>
    </tr>
    <tr>
      <th>8</th>
      <td>mean</td>
      <td>aggregation</td>
      <td>Computes the average value of a numeric feature.</td>
    </tr>
    <tr>
      <th>9</th>
      <td>median</td>
      <td>aggregation</td>
      <td>Finds the median value of any feature with wel...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>all</td>
      <td>aggregation</td>
      <td>Test if all values are 'True'.</td>
    </tr>
    <tr>
      <th>11</th>
      <td>avg_time_between</td>
      <td>aggregation</td>
      <td>Computes the average time between consecutive ...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>percent_true</td>
      <td>aggregation</td>
      <td>Finds the percent of 'True' values in a boolea...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>any</td>
      <td>aggregation</td>
      <td>Test if any value is 'True'.</td>
    </tr>
    <tr>
      <th>14</th>
      <td>n_most_common</td>
      <td>aggregation</td>
      <td>Finds the N most common elements in a categori...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>num_true</td>
      <td>aggregation</td>
      <td>Finds the number of 'True' values in a boolean.</td>
    </tr>
    <tr>
      <th>16</th>
      <td>last</td>
      <td>aggregation</td>
      <td>Returns the last value.</td>
    </tr>
    <tr>
      <th>17</th>
      <td>num_unique</td>
      <td>aggregation</td>
      <td>Returns the number of unique categorical varia...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>std</td>
      <td>aggregation</td>
      <td>Finds the standard deviation of a numeric feat...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>numwords</td>
      <td>transform</td>
      <td>Returns the words in a given string by countin...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>minute</td>
      <td>transform</td>
      <td>Transform a Datetime feature into the minute.</td>
    </tr>
    <tr>
      <th>21</th>
      <td>is_null</td>
      <td>transform</td>
      <td>For each value of base feature, return 'True' ...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>cum_sum</td>
      <td>transform</td>
      <td>Calculates the sum of previous values of an in...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>and</td>
      <td>transform</td>
      <td>For two boolean values, determine if both valu...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>weekday</td>
      <td>transform</td>
      <td>Transform Datetime feature into the boolean of...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>second</td>
      <td>transform</td>
      <td>Transform a Datetime feature into the second.</td>
    </tr>
    <tr>
      <th>26</th>
      <td>days_since</td>
      <td>transform</td>
      <td>For each value of the base feature, compute th...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>longitude</td>
      <td>transform</td>
      <td>Returns the second value on the tuple base fea...</td>
    </tr>
    <tr>
      <th>28</th>
      <td>negate</td>
      <td>transform</td>
      <td>Creates a transform feature that negates a fea...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>years</td>
      <td>transform</td>
      <td>Transform a Timedelta feature into the number ...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>32</th>
      <td>percentile</td>
      <td>transform</td>
      <td>For each value of the base feature, determines...</td>
    </tr>
    <tr>
      <th>33</th>
      <td>divide</td>
      <td>transform</td>
      <td>Creates a transform feature that divides two f...</td>
    </tr>
    <tr>
      <th>34</th>
      <td>months</td>
      <td>transform</td>
      <td>Transform a Timedelta feature into the number ...</td>
    </tr>
    <tr>
      <th>35</th>
      <td>day</td>
      <td>transform</td>
      <td>Transform a Datetime feature into the day.</td>
    </tr>
    <tr>
      <th>36</th>
      <td>cum_count</td>
      <td>transform</td>
      <td>Calculates the number of previous values of an...</td>
    </tr>
    <tr>
      <th>37</th>
      <td>diff</td>
      <td>transform</td>
      <td>Compute the difference between the value of a ...</td>
    </tr>
    <tr>
      <th>38</th>
      <td>subtract</td>
      <td>transform</td>
      <td>Creates a transform feature that subtracts two...</td>
    </tr>
    <tr>
      <th>39</th>
      <td>weeks</td>
      <td>transform</td>
      <td>Transform a Timedelta feature into the number ...</td>
    </tr>
    <tr>
      <th>40</th>
      <td>minutes</td>
      <td>transform</td>
      <td>Transform a Timedelta feature into the number ...</td>
    </tr>
    <tr>
      <th>41</th>
      <td>absolute</td>
      <td>transform</td>
      <td>Absolute value of base feature.</td>
    </tr>
    <tr>
      <th>42</th>
      <td>characters</td>
      <td>transform</td>
      <td>Return the characters in a given string.</td>
    </tr>
    <tr>
      <th>43</th>
      <td>or</td>
      <td>transform</td>
      <td>For two boolean values, determine if one value...</td>
    </tr>
    <tr>
      <th>44</th>
      <td>seconds</td>
      <td>transform</td>
      <td>Transform a Timedelta feature into the number ...</td>
    </tr>
    <tr>
      <th>45</th>
      <td>week</td>
      <td>transform</td>
      <td>Transform a Datetime feature into the week.</td>
    </tr>
    <tr>
      <th>46</th>
      <td>isin</td>
      <td>transform</td>
      <td>For each value of the base feature, checks whe...</td>
    </tr>
    <tr>
      <th>47</th>
      <td>haversine</td>
      <td>transform</td>
      <td>Calculate the approximate haversine distance i...</td>
    </tr>
    <tr>
      <th>48</th>
      <td>weekend</td>
      <td>transform</td>
      <td>Transform Datetime feature into the boolean of...</td>
    </tr>
    <tr>
      <th>49</th>
      <td>hours</td>
      <td>transform</td>
      <td>Transform a Timedelta feature into the number ...</td>
    </tr>
    <tr>
      <th>50</th>
      <td>latitude</td>
      <td>transform</td>
      <td>Returns the first value of the tuple base feat...</td>
    </tr>
    <tr>
      <th>51</th>
      <td>time_since</td>
      <td>transform</td>
      <td>Calculates time since the cutoff time.</td>
    </tr>
    <tr>
      <th>52</th>
      <td>mod</td>
      <td>transform</td>
      <td>Creates a transform feature that divides two f...</td>
    </tr>
    <tr>
      <th>53</th>
      <td>year</td>
      <td>transform</td>
      <td>Transform a Datetime feature into the year.</td>
    </tr>
    <tr>
      <th>54</th>
      <td>days</td>
      <td>transform</td>
      <td>Transform a Timedelta feature into the number ...</td>
    </tr>
    <tr>
      <th>55</th>
      <td>cum_max</td>
      <td>transform</td>
      <td>Calculates the max of previous values of an in...</td>
    </tr>
    <tr>
      <th>56</th>
      <td>not</td>
      <td>transform</td>
      <td>For each value of the base feature, negates th...</td>
    </tr>
    <tr>
      <th>57</th>
      <td>month</td>
      <td>transform</td>
      <td>Transform a Datetime feature into the month.</td>
    </tr>
    <tr>
      <th>58</th>
      <td>multiply</td>
      <td>transform</td>
      <td>Creates a transform feature that multplies two...</td>
    </tr>
    <tr>
      <th>59</th>
      <td>cum_mean</td>
      <td>transform</td>
      <td>Calculates the mean of previous values of an i...</td>
    </tr>
    <tr>
      <th>60</th>
      <td>add</td>
      <td>transform</td>
      <td>Creates a transform feature that adds two feat...</td>
    </tr>
    <tr>
      <th>61</th>
      <td>time_since_previous</td>
      <td>transform</td>
      <td>Compute the time since the previous instance.</td>
    </tr>
  </tbody>
</table>
<p>62 rows × 3 columns</p>
</div>




```python
pd.options.display.max_colwidth = 500
primitives[primitives['type'] == 'aggregation'].head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>type</th>
      <th>description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>trend</td>
      <td>aggregation</td>
      <td>Calculates the slope of the linear trend of variable overtime.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>max</td>
      <td>aggregation</td>
      <td>Finds the maximum non-null value of a numeric feature.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mode</td>
      <td>aggregation</td>
      <td>Finds the most common element in a categorical feature.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>skew</td>
      <td>aggregation</td>
      <td>Computes the skewness of a data set.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>min</td>
      <td>aggregation</td>
      <td>Finds the minimum non-null value of a numeric feature.</td>
    </tr>
    <tr>
      <th>5</th>
      <td>sum</td>
      <td>aggregation</td>
      <td>Counts the number of elements of a numeric or boolean feature.</td>
    </tr>
    <tr>
      <th>6</th>
      <td>time_since_last</td>
      <td>aggregation</td>
      <td>Time since last related instance.</td>
    </tr>
    <tr>
      <th>7</th>
      <td>count</td>
      <td>aggregation</td>
      <td>Counts the number of non null values.</td>
    </tr>
    <tr>
      <th>8</th>
      <td>mean</td>
      <td>aggregation</td>
      <td>Computes the average value of a numeric feature.</td>
    </tr>
    <tr>
      <th>9</th>
      <td>median</td>
      <td>aggregation</td>
      <td>Finds the median value of any feature with well-ordered values.</td>
    </tr>
  </tbody>
</table>
</div>




```python
loans
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>client_id</th>
      <th>loan_type</th>
      <th>loan_amount</th>
      <th>repaid</th>
      <th>loan_id</th>
      <th>loan_start</th>
      <th>loan_end</th>
      <th>rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>46109</td>
      <td>home</td>
      <td>13672</td>
      <td>0</td>
      <td>10243</td>
      <td>2002-04-16</td>
      <td>2003-12-20</td>
      <td>2.15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>46109</td>
      <td>credit</td>
      <td>9794</td>
      <td>0</td>
      <td>10984</td>
      <td>2003-10-21</td>
      <td>2005-07-17</td>
      <td>1.25</td>
    </tr>
    <tr>
      <th>2</th>
      <td>46109</td>
      <td>home</td>
      <td>12734</td>
      <td>1</td>
      <td>10990</td>
      <td>2006-02-01</td>
      <td>2007-07-05</td>
      <td>0.68</td>
    </tr>
    <tr>
      <th>3</th>
      <td>46109</td>
      <td>cash</td>
      <td>12518</td>
      <td>1</td>
      <td>10596</td>
      <td>2010-12-08</td>
      <td>2013-05-05</td>
      <td>1.24</td>
    </tr>
    <tr>
      <th>4</th>
      <td>46109</td>
      <td>credit</td>
      <td>14049</td>
      <td>1</td>
      <td>11415</td>
      <td>2010-07-07</td>
      <td>2012-05-21</td>
      <td>3.13</td>
    </tr>
    <tr>
      <th>5</th>
      <td>46109</td>
      <td>home</td>
      <td>6935</td>
      <td>0</td>
      <td>11501</td>
      <td>2006-09-17</td>
      <td>2008-11-26</td>
      <td>1.94</td>
    </tr>
    <tr>
      <th>6</th>
      <td>46109</td>
      <td>cash</td>
      <td>6177</td>
      <td>1</td>
      <td>11141</td>
      <td>2007-03-12</td>
      <td>2009-04-26</td>
      <td>9.48</td>
    </tr>
    <tr>
      <th>7</th>
      <td>46109</td>
      <td>home</td>
      <td>12656</td>
      <td>0</td>
      <td>11658</td>
      <td>2006-05-26</td>
      <td>2007-10-15</td>
      <td>4.14</td>
    </tr>
    <tr>
      <th>8</th>
      <td>46109</td>
      <td>home</td>
      <td>11062</td>
      <td>1</td>
      <td>11611</td>
      <td>2012-09-12</td>
      <td>2014-03-14</td>
      <td>5.48</td>
    </tr>
    <tr>
      <th>9</th>
      <td>46109</td>
      <td>other</td>
      <td>4050</td>
      <td>1</td>
      <td>10828</td>
      <td>2003-12-06</td>
      <td>2005-08-19</td>
      <td>4.26</td>
    </tr>
    <tr>
      <th>10</th>
      <td>46109</td>
      <td>other</td>
      <td>1618</td>
      <td>0</td>
      <td>11661</td>
      <td>2006-08-28</td>
      <td>2009-04-23</td>
      <td>6.49</td>
    </tr>
    <tr>
      <th>11</th>
      <td>46109</td>
      <td>home</td>
      <td>8406</td>
      <td>0</td>
      <td>11259</td>
      <td>2011-10-22</td>
      <td>2013-06-11</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>12</th>
      <td>46109</td>
      <td>cash</td>
      <td>9057</td>
      <td>1</td>
      <td>10856</td>
      <td>2005-06-17</td>
      <td>2007-03-01</td>
      <td>0.86</td>
    </tr>
    <tr>
      <th>13</th>
      <td>46109</td>
      <td>credit</td>
      <td>3524</td>
      <td>0</td>
      <td>11867</td>
      <td>2005-09-18</td>
      <td>2007-08-27</td>
      <td>5.98</td>
    </tr>
    <tr>
      <th>14</th>
      <td>46109</td>
      <td>other</td>
      <td>10853</td>
      <td>0</td>
      <td>11961</td>
      <td>2013-11-26</td>
      <td>2015-08-06</td>
      <td>2.82</td>
    </tr>
    <tr>
      <th>15</th>
      <td>46109</td>
      <td>credit</td>
      <td>7339</td>
      <td>1</td>
      <td>10328</td>
      <td>2001-09-24</td>
      <td>2003-08-31</td>
      <td>1.29</td>
    </tr>
    <tr>
      <th>16</th>
      <td>46109</td>
      <td>credit</td>
      <td>11953</td>
      <td>1</td>
      <td>11307</td>
      <td>2013-05-15</td>
      <td>2015-03-15</td>
      <td>3.30</td>
    </tr>
    <tr>
      <th>17</th>
      <td>46109</td>
      <td>other</td>
      <td>10067</td>
      <td>1</td>
      <td>10422</td>
      <td>2004-04-05</td>
      <td>2006-10-13</td>
      <td>3.12</td>
    </tr>
    <tr>
      <th>18</th>
      <td>46109</td>
      <td>credit</td>
      <td>12009</td>
      <td>0</td>
      <td>11424</td>
      <td>2001-03-25</td>
      <td>2003-10-04</td>
      <td>0.79</td>
    </tr>
    <tr>
      <th>19</th>
      <td>46109</td>
      <td>credit</td>
      <td>559</td>
      <td>1</td>
      <td>10599</td>
      <td>2008-02-15</td>
      <td>2009-11-25</td>
      <td>4.15</td>
    </tr>
    <tr>
      <th>20</th>
      <td>49545</td>
      <td>other</td>
      <td>13010</td>
      <td>1</td>
      <td>11345</td>
      <td>2009-12-13</td>
      <td>2011-08-30</td>
      <td>1.94</td>
    </tr>
    <tr>
      <th>21</th>
      <td>49545</td>
      <td>cash</td>
      <td>3851</td>
      <td>1</td>
      <td>11033</td>
      <td>2011-04-27</td>
      <td>2014-01-07</td>
      <td>0.81</td>
    </tr>
    <tr>
      <th>22</th>
      <td>49545</td>
      <td>other</td>
      <td>8419</td>
      <td>1</td>
      <td>11054</td>
      <td>2006-10-24</td>
      <td>2009-07-05</td>
      <td>0.83</td>
    </tr>
    <tr>
      <th>23</th>
      <td>49545</td>
      <td>home</td>
      <td>11799</td>
      <td>0</td>
      <td>11030</td>
      <td>2001-04-24</td>
      <td>2003-11-08</td>
      <td>6.52</td>
    </tr>
    <tr>
      <th>24</th>
      <td>49545</td>
      <td>cash</td>
      <td>10422</td>
      <td>1</td>
      <td>10709</td>
      <td>2008-11-04</td>
      <td>2010-09-14</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>25</th>
      <td>49545</td>
      <td>home</td>
      <td>14971</td>
      <td>1</td>
      <td>11247</td>
      <td>2011-07-26</td>
      <td>2013-12-15</td>
      <td>0.15</td>
    </tr>
    <tr>
      <th>26</th>
      <td>49545</td>
      <td>other</td>
      <td>4131</td>
      <td>0</td>
      <td>10939</td>
      <td>2008-10-13</td>
      <td>2010-02-26</td>
      <td>4.07</td>
    </tr>
    <tr>
      <th>27</th>
      <td>49545</td>
      <td>credit</td>
      <td>4458</td>
      <td>1</td>
      <td>10192</td>
      <td>2013-02-11</td>
      <td>2014-09-11</td>
      <td>3.60</td>
    </tr>
    <tr>
      <th>28</th>
      <td>49545</td>
      <td>other</td>
      <td>8172</td>
      <td>1</td>
      <td>11866</td>
      <td>2002-02-06</td>
      <td>2004-10-02</td>
      <td>1.41</td>
    </tr>
    <tr>
      <th>29</th>
      <td>49545</td>
      <td>cash</td>
      <td>14946</td>
      <td>1</td>
      <td>11043</td>
      <td>2002-06-28</td>
      <td>2004-03-17</td>
      <td>5.97</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>413</th>
      <td>49624</td>
      <td>credit</td>
      <td>14679</td>
      <td>1</td>
      <td>11524</td>
      <td>2006-04-03</td>
      <td>2008-04-12</td>
      <td>6.49</td>
    </tr>
    <tr>
      <th>414</th>
      <td>49624</td>
      <td>other</td>
      <td>2572</td>
      <td>1</td>
      <td>10578</td>
      <td>2004-05-04</td>
      <td>2005-12-16</td>
      <td>2.28</td>
    </tr>
    <tr>
      <th>415</th>
      <td>49624</td>
      <td>cash</td>
      <td>8621</td>
      <td>0</td>
      <td>10454</td>
      <td>2014-07-26</td>
      <td>2015-12-29</td>
      <td>3.18</td>
    </tr>
    <tr>
      <th>416</th>
      <td>49624</td>
      <td>cash</td>
      <td>13221</td>
      <td>1</td>
      <td>11581</td>
      <td>2003-05-07</td>
      <td>2006-01-06</td>
      <td>3.94</td>
    </tr>
    <tr>
      <th>417</th>
      <td>49624</td>
      <td>credit</td>
      <td>11219</td>
      <td>1</td>
      <td>11132</td>
      <td>2013-08-12</td>
      <td>2016-03-15</td>
      <td>3.77</td>
    </tr>
    <tr>
      <th>418</th>
      <td>49624</td>
      <td>cash</td>
      <td>11943</td>
      <td>0</td>
      <td>11310</td>
      <td>2002-10-26</td>
      <td>2004-12-01</td>
      <td>6.24</td>
    </tr>
    <tr>
      <th>419</th>
      <td>49624</td>
      <td>other</td>
      <td>2361</td>
      <td>0</td>
      <td>11002</td>
      <td>2006-09-28</td>
      <td>2008-07-22</td>
      <td>4.24</td>
    </tr>
    <tr>
      <th>420</th>
      <td>49624</td>
      <td>credit</td>
      <td>9296</td>
      <td>1</td>
      <td>10714</td>
      <td>2003-04-28</td>
      <td>2005-05-04</td>
      <td>3.44</td>
    </tr>
    <tr>
      <th>421</th>
      <td>49624</td>
      <td>home</td>
      <td>8133</td>
      <td>1</td>
      <td>10312</td>
      <td>2009-03-14</td>
      <td>2011-03-21</td>
      <td>12.62</td>
    </tr>
    <tr>
      <th>422</th>
      <td>49624</td>
      <td>credit</td>
      <td>6050</td>
      <td>0</td>
      <td>11034</td>
      <td>2011-03-24</td>
      <td>2012-12-21</td>
      <td>3.45</td>
    </tr>
    <tr>
      <th>423</th>
      <td>49624</td>
      <td>other</td>
      <td>13015</td>
      <td>1</td>
      <td>10233</td>
      <td>2002-07-25</td>
      <td>2005-03-26</td>
      <td>0.32</td>
    </tr>
    <tr>
      <th>424</th>
      <td>49624</td>
      <td>home</td>
      <td>7876</td>
      <td>1</td>
      <td>11977</td>
      <td>2005-05-15</td>
      <td>2007-05-04</td>
      <td>5.83</td>
    </tr>
    <tr>
      <th>425</th>
      <td>49624</td>
      <td>credit</td>
      <td>2461</td>
      <td>1</td>
      <td>10185</td>
      <td>2012-10-26</td>
      <td>2015-05-30</td>
      <td>4.12</td>
    </tr>
    <tr>
      <th>426</th>
      <td>49624</td>
      <td>other</td>
      <td>14421</td>
      <td>0</td>
      <td>10826</td>
      <td>2004-09-03</td>
      <td>2006-01-31</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>427</th>
      <td>49624</td>
      <td>home</td>
      <td>11750</td>
      <td>0</td>
      <td>10072</td>
      <td>2003-10-17</td>
      <td>2005-11-04</td>
      <td>0.63</td>
    </tr>
    <tr>
      <th>428</th>
      <td>26945</td>
      <td>credit</td>
      <td>6010</td>
      <td>0</td>
      <td>10078</td>
      <td>2005-03-06</td>
      <td>2006-11-29</td>
      <td>2.64</td>
    </tr>
    <tr>
      <th>429</th>
      <td>26945</td>
      <td>credit</td>
      <td>8337</td>
      <td>0</td>
      <td>11580</td>
      <td>2000-12-07</td>
      <td>2002-05-01</td>
      <td>1.23</td>
    </tr>
    <tr>
      <th>430</th>
      <td>26945</td>
      <td>cash</td>
      <td>9249</td>
      <td>1</td>
      <td>11482</td>
      <td>2013-12-24</td>
      <td>2016-05-11</td>
      <td>2.86</td>
    </tr>
    <tr>
      <th>431</th>
      <td>26945</td>
      <td>other</td>
      <td>8899</td>
      <td>1</td>
      <td>10079</td>
      <td>2002-08-19</td>
      <td>2005-02-24</td>
      <td>3.98</td>
    </tr>
    <tr>
      <th>432</th>
      <td>26945</td>
      <td>credit</td>
      <td>1367</td>
      <td>1</td>
      <td>10439</td>
      <td>2013-07-29</td>
      <td>2016-01-12</td>
      <td>3.51</td>
    </tr>
    <tr>
      <th>433</th>
      <td>26945</td>
      <td>cash</td>
      <td>8685</td>
      <td>1</td>
      <td>10315</td>
      <td>2007-11-08</td>
      <td>2009-10-25</td>
      <td>2.76</td>
    </tr>
    <tr>
      <th>434</th>
      <td>26945</td>
      <td>credit</td>
      <td>3510</td>
      <td>0</td>
      <td>10196</td>
      <td>2002-07-14</td>
      <td>2004-08-10</td>
      <td>3.19</td>
    </tr>
    <tr>
      <th>435</th>
      <td>26945</td>
      <td>home</td>
      <td>653</td>
      <td>0</td>
      <td>11230</td>
      <td>2002-08-08</td>
      <td>2004-05-01</td>
      <td>2.95</td>
    </tr>
    <tr>
      <th>436</th>
      <td>26945</td>
      <td>other</td>
      <td>13726</td>
      <td>0</td>
      <td>11442</td>
      <td>2007-05-16</td>
      <td>2010-01-12</td>
      <td>1.20</td>
    </tr>
    <tr>
      <th>437</th>
      <td>26945</td>
      <td>home</td>
      <td>14593</td>
      <td>1</td>
      <td>10812</td>
      <td>2009-10-16</td>
      <td>2011-07-07</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>438</th>
      <td>26945</td>
      <td>other</td>
      <td>12963</td>
      <td>0</td>
      <td>10330</td>
      <td>2001-11-26</td>
      <td>2004-06-11</td>
      <td>2.46</td>
    </tr>
    <tr>
      <th>439</th>
      <td>26945</td>
      <td>credit</td>
      <td>1728</td>
      <td>1</td>
      <td>10248</td>
      <td>2004-01-27</td>
      <td>2005-06-21</td>
      <td>5.27</td>
    </tr>
    <tr>
      <th>440</th>
      <td>26945</td>
      <td>other</td>
      <td>9329</td>
      <td>0</td>
      <td>10154</td>
      <td>2001-12-17</td>
      <td>2004-07-22</td>
      <td>5.65</td>
    </tr>
    <tr>
      <th>441</th>
      <td>26945</td>
      <td>home</td>
      <td>4197</td>
      <td>0</td>
      <td>10333</td>
      <td>2003-10-16</td>
      <td>2005-07-10</td>
      <td>4.50</td>
    </tr>
    <tr>
      <th>442</th>
      <td>26945</td>
      <td>home</td>
      <td>3643</td>
      <td>0</td>
      <td>11434</td>
      <td>2010-03-24</td>
      <td>2011-12-22</td>
      <td>0.13</td>
    </tr>
  </tbody>
</table>
<p>443 rows × 8 columns</p>
</div>




```python
primitives[primitives['type'] == 'transform'].head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>type</th>
      <th>description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19</th>
      <td>numwords</td>
      <td>transform</td>
      <td>Returns the words in a given string by counting the spaces.</td>
    </tr>
    <tr>
      <th>20</th>
      <td>minute</td>
      <td>transform</td>
      <td>Transform a Datetime feature into the minute.</td>
    </tr>
    <tr>
      <th>21</th>
      <td>is_null</td>
      <td>transform</td>
      <td>For each value of base feature, return 'True' if value is null.</td>
    </tr>
    <tr>
      <th>22</th>
      <td>cum_sum</td>
      <td>transform</td>
      <td>Calculates the sum of previous values of an instance for each value in a time-dependent entity.</td>
    </tr>
    <tr>
      <th>23</th>
      <td>and</td>
      <td>transform</td>
      <td>For two boolean values, determine if both values are 'True'.</td>
    </tr>
    <tr>
      <th>24</th>
      <td>weekday</td>
      <td>transform</td>
      <td>Transform Datetime feature into the boolean of Weekday.</td>
    </tr>
    <tr>
      <th>25</th>
      <td>second</td>
      <td>transform</td>
      <td>Transform a Datetime feature into the second.</td>
    </tr>
    <tr>
      <th>26</th>
      <td>days_since</td>
      <td>transform</td>
      <td>For each value of the base feature, compute the number of days between it</td>
    </tr>
    <tr>
      <th>27</th>
      <td>longitude</td>
      <td>transform</td>
      <td>Returns the second value on the tuple base feature.</td>
    </tr>
    <tr>
      <th>28</th>
      <td>negate</td>
      <td>transform</td>
      <td>Creates a transform feature that negates a feature.</td>
    </tr>
  </tbody>
</table>
</div>



If featuretools does not have enough primitives for us, we can [also make our own.](https://docs.featuretools.com/automated_feature_engineering/primitives.html#defining-custom-primitives) 

To get an idea of what a feature primitive actually does, let's try out a few on our data. Using primitives is surprisingly easy using the `ft.dfs` function (which stands for deep feature synthesis). In this function, we specify the entityset to use; the `target_entity`, which is the dataframe we want to make the features for (where the features end up); the `agg_primitives` which are the aggregation feature primitives; and the `trans_primitives` which are the transformation primitives to apply. 

In the following example, we are using the `EntitySet` we already created, the target entity is the `clients` dataframe because we want to make new features about each client, and then we specify a few aggregation and transformation primitives. 


```python
# Create new features using specified primitives
features, feature_names = ft.dfs(entityset = es, target_entity = 'clients', 
                                 agg_primitives = ['mean', 'max', 'percent_true', 'last'],
                                 trans_primitives = ['years', 'month', 'subtract', 'divide'])
```


```python
features
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>income</th>
      <th>credit_score</th>
      <th>join_month</th>
      <th>log_income</th>
      <th>MEAN(loans.loan_amount)</th>
      <th>MEAN(loans.rate)</th>
      <th>MAX(loans.loan_amount)</th>
      <th>MAX(loans.rate)</th>
      <th>LAST(loans.loan_type)</th>
      <th>LAST(loans.loan_amount)</th>
      <th>...</th>
      <th>join_month - log_income / income - join_month</th>
      <th>join_month - income / join_month - credit_score</th>
      <th>MAX(loans.loan_amount) / join_month</th>
      <th>LAST(loans.rate) / credit_score - join_month</th>
      <th>MEAN(payments.payment_amount) / income - join_month</th>
      <th>join_month - log_income / MAX(payments.payment_amount)</th>
      <th>join_month - income / income - join_month</th>
      <th>MAX(loans.rate) / credit_score - join_month</th>
      <th>MAX(loans.loan_amount) / log_income - income</th>
      <th>LAST(loans.loan_amount) / MEAN(loans.rate)</th>
    </tr>
    <tr>
      <th>client_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25707</th>
      <td>211422</td>
      <td>621</td>
      <td>10</td>
      <td>12.261611</td>
      <td>7963.950000</td>
      <td>3.477000</td>
      <td>13913</td>
      <td>9.44</td>
      <td>home</td>
      <td>2203</td>
      <td>...</td>
      <td>-0.000011</td>
      <td>346.009820</td>
      <td>1391.300000</td>
      <td>0.012111</td>
      <td>0.005575</td>
      <td>-0.000836</td>
      <td>-1.0</td>
      <td>0.015450</td>
      <td>-0.065811</td>
      <td>633.592177</td>
    </tr>
    <tr>
      <th>26326</th>
      <td>227920</td>
      <td>633</td>
      <td>5</td>
      <td>12.336750</td>
      <td>7270.062500</td>
      <td>2.517500</td>
      <td>13464</td>
      <td>6.73</td>
      <td>credit</td>
      <td>5275</td>
      <td>...</td>
      <td>-0.000032</td>
      <td>362.921975</td>
      <td>2692.800000</td>
      <td>0.002309</td>
      <td>0.005119</td>
      <td>-0.002760</td>
      <td>-1.0</td>
      <td>0.010717</td>
      <td>-0.059077</td>
      <td>2095.332671</td>
    </tr>
    <tr>
      <th>26695</th>
      <td>174532</td>
      <td>680</td>
      <td>8</td>
      <td>12.069863</td>
      <td>7824.722222</td>
      <td>2.466111</td>
      <td>14865</td>
      <td>6.51</td>
      <td>other</td>
      <td>13918</td>
      <td>...</td>
      <td>-0.000023</td>
      <td>259.708333</td>
      <td>1858.125000</td>
      <td>0.001339</td>
      <td>0.006918</td>
      <td>-0.001388</td>
      <td>-1.0</td>
      <td>0.009687</td>
      <td>-0.085177</td>
      <td>5643.703537</td>
    </tr>
    <tr>
      <th>26945</th>
      <td>214516</td>
      <td>806</td>
      <td>11</td>
      <td>12.276140</td>
      <td>7125.933333</td>
      <td>2.855333</td>
      <td>14593</td>
      <td>5.65</td>
      <td>cash</td>
      <td>9249</td>
      <td>...</td>
      <td>-0.000006</td>
      <td>269.817610</td>
      <td>1326.636364</td>
      <td>0.003597</td>
      <td>0.005172</td>
      <td>-0.000461</td>
      <td>-1.0</td>
      <td>0.007107</td>
      <td>-0.068031</td>
      <td>3239.201494</td>
    </tr>
    <tr>
      <th>29841</th>
      <td>38354</td>
      <td>523</td>
      <td>8</td>
      <td>10.554614</td>
      <td>9813.000000</td>
      <td>3.445000</td>
      <td>14837</td>
      <td>6.76</td>
      <td>home</td>
      <td>7223</td>
      <td>...</td>
      <td>-0.000067</td>
      <td>74.458252</td>
      <td>1854.625000</td>
      <td>0.009883</td>
      <td>0.037538</td>
      <td>-0.000882</td>
      <td>-1.0</td>
      <td>0.013126</td>
      <td>-0.386950</td>
      <td>2096.661829</td>
    </tr>
    <tr>
      <th>32726</th>
      <td>235705</td>
      <td>730</td>
      <td>5</td>
      <td>12.370336</td>
      <td>6633.263158</td>
      <td>3.058947</td>
      <td>14802</td>
      <td>9.10</td>
      <td>other</td>
      <td>5325</td>
      <td>...</td>
      <td>-0.000031</td>
      <td>325.103448</td>
      <td>2960.400000</td>
      <td>0.003903</td>
      <td>0.004006</td>
      <td>-0.002735</td>
      <td>-1.0</td>
      <td>0.012552</td>
      <td>-0.062802</td>
      <td>1740.794907</td>
    </tr>
    <tr>
      <th>32885</th>
      <td>58955</td>
      <td>642</td>
      <td>5</td>
      <td>10.984530</td>
      <td>9920.400000</td>
      <td>2.436000</td>
      <td>14162</td>
      <td>9.11</td>
      <td>other</td>
      <td>11886</td>
      <td>...</td>
      <td>-0.000102</td>
      <td>92.543171</td>
      <td>2832.400000</td>
      <td>0.014301</td>
      <td>0.023689</td>
      <td>-0.002471</td>
      <td>-1.0</td>
      <td>0.014301</td>
      <td>-0.240262</td>
      <td>4879.310345</td>
    </tr>
    <tr>
      <th>32961</th>
      <td>230341</td>
      <td>714</td>
      <td>4</td>
      <td>12.347316</td>
      <td>7882.235294</td>
      <td>3.930588</td>
      <td>14784</td>
      <td>9.14</td>
      <td>cash</td>
      <td>1693</td>
      <td>...</td>
      <td>-0.000036</td>
      <td>324.418310</td>
      <td>3696.000000</td>
      <td>0.002056</td>
      <td>0.004511</td>
      <td>-0.002991</td>
      <td>-1.0</td>
      <td>0.012873</td>
      <td>-0.064187</td>
      <td>430.724334</td>
    </tr>
    <tr>
      <th>35089</th>
      <td>131176</td>
      <td>771</td>
      <td>3</td>
      <td>11.784295</td>
      <td>6939.200000</td>
      <td>3.513500</td>
      <td>13194</td>
      <td>7.63</td>
      <td>other</td>
      <td>773</td>
      <td>...</td>
      <td>-0.000067</td>
      <td>170.798177</td>
      <td>4398.000000</td>
      <td>0.009935</td>
      <td>0.008346</td>
      <td>-0.003427</td>
      <td>-1.0</td>
      <td>0.009935</td>
      <td>-0.100591</td>
      <td>220.008538</td>
    </tr>
    <tr>
      <th>35214</th>
      <td>95849</td>
      <td>696</td>
      <td>8</td>
      <td>11.470529</td>
      <td>7173.555556</td>
      <td>3.108333</td>
      <td>14767</td>
      <td>8.44</td>
      <td>home</td>
      <td>9389</td>
      <td>...</td>
      <td>-0.000036</td>
      <td>139.303779</td>
      <td>1845.875000</td>
      <td>0.002035</td>
      <td>0.011237</td>
      <td>-0.001208</td>
      <td>-1.0</td>
      <td>0.012267</td>
      <td>-0.154084</td>
      <td>3020.589812</td>
    </tr>
    <tr>
      <th>38537</th>
      <td>127183</td>
      <td>643</td>
      <td>10</td>
      <td>11.753382</td>
      <td>8986.352941</td>
      <td>2.389412</td>
      <td>14804</td>
      <td>8.01</td>
      <td>cash</td>
      <td>8797</td>
      <td>...</td>
      <td>-0.000014</td>
      <td>200.905213</td>
      <td>1480.400000</td>
      <td>0.001011</td>
      <td>0.010542</td>
      <td>-0.000607</td>
      <td>-1.0</td>
      <td>0.012654</td>
      <td>-0.116410</td>
      <td>3681.659281</td>
    </tr>
    <tr>
      <th>39384</th>
      <td>191204</td>
      <td>617</td>
      <td>6</td>
      <td>12.161096</td>
      <td>7865.473684</td>
      <td>3.538421</td>
      <td>14654</td>
      <td>9.23</td>
      <td>other</td>
      <td>14654</td>
      <td>...</td>
      <td>-0.000032</td>
      <td>312.926350</td>
      <td>2442.333333</td>
      <td>0.003699</td>
      <td>0.006243</td>
      <td>-0.002183</td>
      <td>-1.0</td>
      <td>0.015106</td>
      <td>-0.076646</td>
      <td>4141.395210</td>
    </tr>
    <tr>
      <th>39505</th>
      <td>153873</td>
      <td>610</td>
      <td>10</td>
      <td>11.943883</td>
      <td>7424.050000</td>
      <td>3.190500</td>
      <td>14575</td>
      <td>9.91</td>
      <td>cash</td>
      <td>9600</td>
      <td>...</td>
      <td>-0.000013</td>
      <td>256.438333</td>
      <td>1457.500000</td>
      <td>0.000417</td>
      <td>0.007552</td>
      <td>-0.000703</td>
      <td>-1.0</td>
      <td>0.016517</td>
      <td>-0.094728</td>
      <td>3008.932769</td>
    </tr>
    <tr>
      <th>41472</th>
      <td>152214</td>
      <td>638</td>
      <td>11</td>
      <td>11.933043</td>
      <td>7510.812500</td>
      <td>3.981250</td>
      <td>13657</td>
      <td>9.82</td>
      <td>cash</td>
      <td>10122</td>
      <td>...</td>
      <td>-0.000006</td>
      <td>242.748006</td>
      <td>1241.545455</td>
      <td>0.001643</td>
      <td>0.007418</td>
      <td>-0.000383</td>
      <td>-1.0</td>
      <td>0.015662</td>
      <td>-0.089729</td>
      <td>2542.417582</td>
    </tr>
    <tr>
      <th>41480</th>
      <td>122607</td>
      <td>585</td>
      <td>3</td>
      <td>11.716739</td>
      <td>7894.850000</td>
      <td>3.110500</td>
      <td>14399</td>
      <td>10.49</td>
      <td>home</td>
      <td>11154</td>
      <td>...</td>
      <td>-0.000071</td>
      <td>210.659794</td>
      <td>4799.666667</td>
      <td>0.011134</td>
      <td>0.010167</td>
      <td>-0.003206</td>
      <td>-1.0</td>
      <td>0.018024</td>
      <td>-0.117452</td>
      <td>3585.918663</td>
    </tr>
    <tr>
      <th>42320</th>
      <td>229481</td>
      <td>563</td>
      <td>4</td>
      <td>12.343576</td>
      <td>7062.066667</td>
      <td>2.457333</td>
      <td>13887</td>
      <td>6.74</td>
      <td>home</td>
      <td>8090</td>
      <td>...</td>
      <td>-0.000036</td>
      <td>410.513417</td>
      <td>3471.750000</td>
      <td>0.005689</td>
      <td>0.004451</td>
      <td>-0.003013</td>
      <td>-1.0</td>
      <td>0.012057</td>
      <td>-0.060518</td>
      <td>3292.186652</td>
    </tr>
    <tr>
      <th>44387</th>
      <td>151903</td>
      <td>781</td>
      <td>7</td>
      <td>11.930997</td>
      <td>7387.733333</td>
      <td>2.806667</td>
      <td>14900</td>
      <td>5.97</td>
      <td>credit</td>
      <td>7279</td>
      <td>...</td>
      <td>-0.000032</td>
      <td>196.248062</td>
      <td>2128.571429</td>
      <td>0.005297</td>
      <td>0.006950</td>
      <td>-0.001692</td>
      <td>-1.0</td>
      <td>0.007713</td>
      <td>-0.098097</td>
      <td>2593.467933</td>
    </tr>
    <tr>
      <th>44601</th>
      <td>156341</td>
      <td>518</td>
      <td>10</td>
      <td>11.959795</td>
      <td>6869.250000</td>
      <td>4.151000</td>
      <td>14104</td>
      <td>9.22</td>
      <td>other</td>
      <td>6129</td>
      <td>...</td>
      <td>-0.000013</td>
      <td>307.738189</td>
      <td>1410.400000</td>
      <td>0.005433</td>
      <td>0.006958</td>
      <td>-0.000737</td>
      <td>-1.0</td>
      <td>0.018150</td>
      <td>-0.090220</td>
      <td>1476.511684</td>
    </tr>
    <tr>
      <th>46109</th>
      <td>172677</td>
      <td>527</td>
      <td>4</td>
      <td>12.059178</td>
      <td>8951.600000</td>
      <td>3.152500</td>
      <td>14049</td>
      <td>9.48</td>
      <td>other</td>
      <td>10853</td>
      <td>...</td>
      <td>-0.000047</td>
      <td>330.158700</td>
      <td>3512.250000</td>
      <td>0.005392</td>
      <td>0.007966</td>
      <td>-0.002947</td>
      <td>-1.0</td>
      <td>0.018126</td>
      <td>-0.081366</td>
      <td>3442.664552</td>
    </tr>
    <tr>
      <th>46180</th>
      <td>43851</td>
      <td>562</td>
      <td>11</td>
      <td>10.688553</td>
      <td>7700.850000</td>
      <td>3.502500</td>
      <td>14081</td>
      <td>9.26</td>
      <td>other</td>
      <td>3834</td>
      <td>...</td>
      <td>0.000007</td>
      <td>79.564428</td>
      <td>1280.090909</td>
      <td>0.002505</td>
      <td>0.027065</td>
      <td>0.000117</td>
      <td>-1.0</td>
      <td>0.016806</td>
      <td>-0.321188</td>
      <td>1094.646681</td>
    </tr>
    <tr>
      <th>46958</th>
      <td>225709</td>
      <td>644</td>
      <td>7</td>
      <td>12.327002</td>
      <td>9378.384615</td>
      <td>3.153846</td>
      <td>14942</td>
      <td>6.17</td>
      <td>credit</td>
      <td>8308</td>
      <td>...</td>
      <td>-0.000024</td>
      <td>354.320251</td>
      <td>2134.571429</td>
      <td>0.000502</td>
      <td>0.005965</td>
      <td>-0.001784</td>
      <td>-1.0</td>
      <td>0.009686</td>
      <td>-0.066204</td>
      <td>2634.243902</td>
    </tr>
    <tr>
      <th>48177</th>
      <td>190632</td>
      <td>769</td>
      <td>6</td>
      <td>12.158100</td>
      <td>7424.368421</td>
      <td>3.938947</td>
      <td>14740</td>
      <td>10.89</td>
      <td>credit</td>
      <td>659</td>
      <td>...</td>
      <td>-0.000032</td>
      <td>249.837484</td>
      <td>2456.666667</td>
      <td>0.001992</td>
      <td>0.006294</td>
      <td>-0.002261</td>
      <td>-1.0</td>
      <td>0.014273</td>
      <td>-0.077327</td>
      <td>167.303581</td>
    </tr>
    <tr>
      <th>49068</th>
      <td>128813</td>
      <td>603</td>
      <td>2</td>
      <td>11.766117</td>
      <td>7617.888889</td>
      <td>3.095556</td>
      <td>13910</td>
      <td>6.78</td>
      <td>other</td>
      <td>10082</td>
      <td>...</td>
      <td>-0.000076</td>
      <td>214.327787</td>
      <td>6955.000000</td>
      <td>0.001048</td>
      <td>0.009349</td>
      <td>-0.003567</td>
      <td>-1.0</td>
      <td>0.011281</td>
      <td>-0.107996</td>
      <td>3256.927495</td>
    </tr>
    <tr>
      <th>49545</th>
      <td>104564</td>
      <td>770</td>
      <td>11</td>
      <td>11.557555</td>
      <td>10289.300000</td>
      <td>2.684000</td>
      <td>14971</td>
      <td>6.52</td>
      <td>home</td>
      <td>7061</td>
      <td>...</td>
      <td>-0.000005</td>
      <td>137.750988</td>
      <td>1361.000000</td>
      <td>0.005573</td>
      <td>0.014167</td>
      <td>-0.000194</td>
      <td>-1.0</td>
      <td>0.008590</td>
      <td>-0.143191</td>
      <td>2630.774963</td>
    </tr>
    <tr>
      <th>49624</th>
      <td>49036</td>
      <td>800</td>
      <td>8</td>
      <td>10.800310</td>
      <td>9174.533333</td>
      <td>4.037333</td>
      <td>14679</td>
      <td>12.62</td>
      <td>cash</td>
      <td>8621</td>
      <td>...</td>
      <td>-0.000057</td>
      <td>61.904040</td>
      <td>1834.875000</td>
      <td>0.004015</td>
      <td>0.028945</td>
      <td>-0.000977</td>
      <td>-1.0</td>
      <td>0.015934</td>
      <td>-0.299417</td>
      <td>2135.320343</td>
    </tr>
  </tbody>
</table>
<p>25 rows × 797 columns</p>
</div>




```python
len(feature_names)
```




    797




```python
pd.DataFrame(features['MONTH(joined)'].head())
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MONTH(joined)</th>
    </tr>
    <tr>
      <th>client_id</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25707</th>
      <td>10</td>
    </tr>
    <tr>
      <th>26326</th>
      <td>5</td>
    </tr>
    <tr>
      <th>26695</th>
      <td>8</td>
    </tr>
    <tr>
      <th>26945</th>
      <td>11</td>
    </tr>
    <tr>
      <th>29841</th>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.DataFrame(features['MEAN(payments.payment_amount)'].head())
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MEAN(payments.payment_amount)</th>
    </tr>
    <tr>
      <th>client_id</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25707</th>
      <td>1178.552795</td>
    </tr>
    <tr>
      <th>26326</th>
      <td>1166.736842</td>
    </tr>
    <tr>
      <th>26695</th>
      <td>1207.433824</td>
    </tr>
    <tr>
      <th>26945</th>
      <td>1109.473214</td>
    </tr>
    <tr>
      <th>29841</th>
      <td>1439.433333</td>
    </tr>
  </tbody>
</table>
</div>




```python
features.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>income</th>
      <th>credit_score</th>
      <th>join_month</th>
      <th>log_income</th>
      <th>MEAN(loans.loan_amount)</th>
      <th>MEAN(loans.rate)</th>
      <th>MAX(loans.loan_amount)</th>
      <th>MAX(loans.rate)</th>
      <th>LAST(loans.loan_type)</th>
      <th>LAST(loans.loan_amount)</th>
      <th>...</th>
      <th>log_income - income / MAX(payments.payment_amount)</th>
      <th>LAST(loans.loan_amount) / income</th>
      <th>log_income - credit_score / join_month</th>
      <th>join_month - log_income / MEAN(loans.loan_amount)</th>
      <th>join_month / income - join_month</th>
      <th>credit_score - log_income / LAST(loans.rate)</th>
      <th>join_month - log_income / MAX(loans.loan_amount)</th>
      <th>credit_score - income / join_month - log_income</th>
      <th>join_month - income / MEAN(payments.payment_amount)</th>
      <th>join_month - credit_score / log_income - credit_score</th>
    </tr>
    <tr>
      <th>client_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25707</th>
      <td>211422</td>
      <td>621</td>
      <td>10</td>
      <td>12.261611</td>
      <td>7963.950000</td>
      <td>3.477000</td>
      <td>13913</td>
      <td>9.44</td>
      <td>home</td>
      <td>2203</td>
      <td>...</td>
      <td>-78.184075</td>
      <td>0.010420</td>
      <td>-60.873839</td>
      <td>-0.000284</td>
      <td>0.000047</td>
      <td>82.261944</td>
      <td>-0.000163</td>
      <td>93208.319781</td>
      <td>-179.382715</td>
      <td>1.003715</td>
    </tr>
    <tr>
      <th>26326</th>
      <td>227920</td>
      <td>633</td>
      <td>5</td>
      <td>12.336750</td>
      <td>7270.062500</td>
      <td>2.517500</td>
      <td>13464</td>
      <td>6.73</td>
      <td>credit</td>
      <td>5275</td>
      <td>...</td>
      <td>-85.744042</td>
      <td>0.023144</td>
      <td>-124.132650</td>
      <td>-0.001009</td>
      <td>0.000022</td>
      <td>428.043621</td>
      <td>-0.000545</td>
      <td>30979.248435</td>
      <td>-195.343964</td>
      <td>1.011821</td>
    </tr>
    <tr>
      <th>26695</th>
      <td>174532</td>
      <td>680</td>
      <td>8</td>
      <td>12.069863</td>
      <td>7824.722222</td>
      <td>2.466111</td>
      <td>14865</td>
      <td>6.51</td>
      <td>other</td>
      <td>13918</td>
      <td>...</td>
      <td>-59.522486</td>
      <td>0.079745</td>
      <td>-83.491267</td>
      <td>-0.000520</td>
      <td>0.000046</td>
      <td>742.144596</td>
      <td>-0.000274</td>
      <td>42716.912967</td>
      <td>-144.541255</td>
      <td>1.006093</td>
    </tr>
    <tr>
      <th>26945</th>
      <td>214516</td>
      <td>806</td>
      <td>11</td>
      <td>12.276140</td>
      <td>7125.933333</td>
      <td>2.855333</td>
      <td>14593</td>
      <td>5.65</td>
      <td>cash</td>
      <td>9249</td>
      <td>...</td>
      <td>-77.494120</td>
      <td>0.043116</td>
      <td>-72.156715</td>
      <td>-0.000179</td>
      <td>0.000051</td>
      <td>277.525825</td>
      <td>-0.000087</td>
      <td>167466.003631</td>
      <td>-193.339503</td>
      <td>1.001608</td>
    </tr>
    <tr>
      <th>29841</th>
      <td>38354</td>
      <td>523</td>
      <td>8</td>
      <td>10.554614</td>
      <td>9813.000000</td>
      <td>3.445000</td>
      <td>14837</td>
      <td>6.76</td>
      <td>home</td>
      <td>7223</td>
      <td>...</td>
      <td>-13.231003</td>
      <td>0.188325</td>
      <td>-64.055673</td>
      <td>-0.000260</td>
      <td>0.000209</td>
      <td>100.676893</td>
      <td>-0.000172</td>
      <td>14808.890291</td>
      <td>-26.639650</td>
      <td>1.004985</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 797 columns</p>
</div>



Already we can see how useful featuretools is: it performed the same operations we did manually but also many more in addition. Examining the names of the features in the dataframe brings us to the final piece of the puzzle: deep features.

## Deep Feature Synthesis

While feature primitives are useful by themselves, the main benefit of using featuretools arises when we stack primitives to get deep features. The depth of a feature is simply the number of primitives required to make a feature. So, a feature that relies on a single aggregation would be a deep feature with a depth of 1, a feature that stacks two primitives would have a depth of 2 and so on. The idea itself is lot simpler than the name "deep feature synthesis" implies. (I think the authors were trying to ride the way of deep neural network hype when they named the method!) To read more about deep feature synthesis, check out [the documentation](https://docs.featuretools.com/automated_feature_engineering/afe.html) or the [original paper by Max Kanter et al](http://www.jmaxkanter.com/static/papers/DSAA_DSM_2015.pdf). 

Already in the dataframe we made by specifying the primitives manually we can see the idea of feature depth. For instance, the MEAN(loans.loan_amount) feature has a depth of 1 because it is made by applying a single aggregation primitive. This feature represents the average size of a client's previous loans.


```python
# Show a feature with a depth of 1
pd.DataFrame(features['MEAN(loans.loan_amount)'].head(10))
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MEAN(loans.loan_amount)</th>
    </tr>
    <tr>
      <th>client_id</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25707</th>
      <td>7963.950000</td>
    </tr>
    <tr>
      <th>26326</th>
      <td>7270.062500</td>
    </tr>
    <tr>
      <th>26695</th>
      <td>7824.722222</td>
    </tr>
    <tr>
      <th>26945</th>
      <td>7125.933333</td>
    </tr>
    <tr>
      <th>29841</th>
      <td>9813.000000</td>
    </tr>
    <tr>
      <th>32726</th>
      <td>6633.263158</td>
    </tr>
    <tr>
      <th>32885</th>
      <td>9920.400000</td>
    </tr>
    <tr>
      <th>32961</th>
      <td>7882.235294</td>
    </tr>
    <tr>
      <th>35089</th>
      <td>6939.200000</td>
    </tr>
    <tr>
      <th>35214</th>
      <td>7173.555556</td>
    </tr>
  </tbody>
</table>
</div>



As well scroll through the features, we see a number of features with a depth of 2. For example, the LAST(loans.(MEAN(payments.payment_amount))) has depth = 2 because it is made by stacking two feature primitives, first an aggregation and then a transformation. This feature represents the average payment amount for the last (most recent) loan for each client.


```python
# Show a feature with a depth of 2
pd.DataFrame(features['LAST(loans.MEAN(payments.payment_amount))'].head(10))
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LAST(loans.MEAN(payments.payment_amount))</th>
    </tr>
    <tr>
      <th>client_id</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25707</th>
      <td>293.500000</td>
    </tr>
    <tr>
      <th>26326</th>
      <td>977.375000</td>
    </tr>
    <tr>
      <th>26695</th>
      <td>1769.166667</td>
    </tr>
    <tr>
      <th>26945</th>
      <td>1598.666667</td>
    </tr>
    <tr>
      <th>29841</th>
      <td>1125.500000</td>
    </tr>
    <tr>
      <th>32726</th>
      <td>799.500000</td>
    </tr>
    <tr>
      <th>32885</th>
      <td>1729.000000</td>
    </tr>
    <tr>
      <th>32961</th>
      <td>282.600000</td>
    </tr>
    <tr>
      <th>35089</th>
      <td>110.400000</td>
    </tr>
    <tr>
      <th>35214</th>
      <td>1410.250000</td>
    </tr>
  </tbody>
</table>
</div>



We can create features of arbitrary depth by stacking more primitives. However, when I have used featuretools I've never gone beyond a depth of 2! After this point, the features become very convoluted to understand. I'd encourage anyone interested to experiment with increasing the depth (maybe for a real problem) and see if there is value to "going deeper".

## Automated Deep Feature Synthesis

In addition to manually specifying aggregation and transformation feature primitives, we can let featuretools automatically generate many new features. We do this by making the same `ft.dfs` function call, but without passing in any primitives. We just set the `max_depth` parameter and featuretools will automatically try many all combinations of feature primitives to the ordered depth. 

When running on large datasets, this process can take quite a while, but for our example data, it will be relatively quick. For this call, we only need to specify the `entityset`, the `target_entity` (which will again be `clients`), and the `max_depth`. 


```python
# Perform deep feature synthesis without specifying primitives
features, feature_names = ft.dfs(entityset=es, target_entity='clients', 
                                 max_depth = 2)
```


```python
features.iloc[:, 4:].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SUM(loans.loan_amount)</th>
      <th>SUM(loans.rate)</th>
      <th>STD(loans.loan_amount)</th>
      <th>STD(loans.rate)</th>
      <th>MAX(loans.loan_amount)</th>
      <th>MAX(loans.rate)</th>
      <th>SKEW(loans.loan_amount)</th>
      <th>SKEW(loans.rate)</th>
      <th>MIN(loans.loan_amount)</th>
      <th>MIN(loans.rate)</th>
      <th>...</th>
      <th>NUM_UNIQUE(loans.WEEKDAY(loan_end))</th>
      <th>MODE(loans.MODE(payments.missed))</th>
      <th>MODE(loans.DAY(loan_start))</th>
      <th>MODE(loans.DAY(loan_end))</th>
      <th>MODE(loans.YEAR(loan_start))</th>
      <th>MODE(loans.YEAR(loan_end))</th>
      <th>MODE(loans.MONTH(loan_start))</th>
      <th>MODE(loans.MONTH(loan_end))</th>
      <th>MODE(loans.WEEKDAY(loan_start))</th>
      <th>MODE(loans.WEEKDAY(loan_end))</th>
    </tr>
    <tr>
      <th>client_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25707</th>
      <td>159279</td>
      <td>69.54</td>
      <td>4044.418728</td>
      <td>2.421285</td>
      <td>13913</td>
      <td>9.44</td>
      <td>-0.172074</td>
      <td>0.679118</td>
      <td>1212</td>
      <td>0.33</td>
      <td>...</td>
      <td>6</td>
      <td>0</td>
      <td>27</td>
      <td>1</td>
      <td>2010</td>
      <td>2007</td>
      <td>1</td>
      <td>8</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26326</th>
      <td>116321</td>
      <td>40.28</td>
      <td>4254.149422</td>
      <td>1.991819</td>
      <td>13464</td>
      <td>6.73</td>
      <td>0.135246</td>
      <td>1.067853</td>
      <td>1164</td>
      <td>0.50</td>
      <td>...</td>
      <td>5</td>
      <td>0</td>
      <td>6</td>
      <td>6</td>
      <td>2003</td>
      <td>2005</td>
      <td>4</td>
      <td>7</td>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>26695</th>
      <td>140845</td>
      <td>44.39</td>
      <td>4078.228493</td>
      <td>1.517660</td>
      <td>14865</td>
      <td>6.51</td>
      <td>0.154467</td>
      <td>0.820060</td>
      <td>2389</td>
      <td>0.22</td>
      <td>...</td>
      <td>6</td>
      <td>0</td>
      <td>3</td>
      <td>14</td>
      <td>2003</td>
      <td>2005</td>
      <td>9</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>26945</th>
      <td>106889</td>
      <td>42.83</td>
      <td>4389.555657</td>
      <td>1.564795</td>
      <td>14593</td>
      <td>5.65</td>
      <td>0.156534</td>
      <td>-0.001998</td>
      <td>653</td>
      <td>0.13</td>
      <td>...</td>
      <td>6</td>
      <td>0</td>
      <td>16</td>
      <td>1</td>
      <td>2002</td>
      <td>2004</td>
      <td>12</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29841</th>
      <td>176634</td>
      <td>62.01</td>
      <td>4090.630609</td>
      <td>2.063092</td>
      <td>14837</td>
      <td>6.76</td>
      <td>-0.212397</td>
      <td>0.050600</td>
      <td>2778</td>
      <td>0.26</td>
      <td>...</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>15</td>
      <td>2005</td>
      <td>2007</td>
      <td>3</td>
      <td>2</td>
      <td>5</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 90 columns</p>
</div>



Deep feature synthesis has created 90 new features out of the existing data! While we could have created all of these manually, I am glad to not have to write all that code by hand. The primary benefit of featuretools is that it creates features without any subjective human biases. Even a human with considerable domain knowledge will be limited by their imagination when making new features (not to mention time). Automated feature engineering is not limited by these factors (instead it's limited by computation time) and provides a good starting point for feature creation. This process likely will not remove the human contribution to feature engineering completely because a human can still use domain knowledge and machine learning expertise to select the most important features or build new features from those suggested by automated deep feature synthesis.

# Next Steps

While automatic feature engineering solves one problem, it provides us with another problem: too many features! Although it's difficult to say which features will be important to a given machine learning task ahead of time, it's likely that not all of the features made by featuretools add value. In fact, having too many features is a significant issue in machine learning because it makes training a model much harder. The [irrelevant features can drown out the important features](https://pdfs.semanticscholar.org/a83b/ddb34618cc68f1014ca12eef7f537825d104.pdf), leaving a model unable to learn how to map the features to the target.

This problem is known as the ["curse of dimensionality"](https://en.wikipedia.org/wiki/Curse_of_dimensionality#Machine_learning) and is addressed through the process of [feature reduction and selection](http://scikit-learn.org/stable/modules/feature_selection.html), which means [removing low-value features](https://machinelearningmastery.com/feature-selection-machine-learning-python/) from the data. Defining which features are useful is an important problem where a data scientist can still add considerable value to the feature engineering task. Feature reduction will have to be another topic for another day!

# Conclusions

In this notebook, we saw how to apply automated feature engineering to an example dataset. This is a powerful method which allows us to overcome the human limits of time and imagination to create many new features from multiple tables of data. Featuretools is built on the idea of deep feature synthesis, which means stacking multiple simple feature primitives - __aggregations and transformations__ - to create new features. Feature engineering allows us to combine information across many tables into a single dataframe that we can then use for machine learning model training. Finally, the next step after creating all of these features is figuring out which ones are important. 

Featuretools is currently the only Python option for this process, but with the recent emphasis on automating aspects of the machine learning pipeline, other competitiors will probably enter the sphere. While the exact tools will change, the idea of automatically creating new features out of existing data will grow in importance. Staying up-to-date on methods such as automated feature engineering is crucial in the rapidly changing field of data science. Now go out there and find a problem on which to apply featuretools! 

For more information, check out the [documentation for featuretools](https://docs.featuretools.com/index.html). Also, read about how featuretools is [used in the real world by Feature Labs](https://www.featurelabs.com/), the company behind the open-source library.
