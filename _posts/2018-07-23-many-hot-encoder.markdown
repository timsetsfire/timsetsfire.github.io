---
layout: post
title:  "Many Hot Encoder"
date:   2018-07-23 23:05:00 -0400
categories: jekyll update
---

source /home/tim/.rvm/scripts/rvm

This post explores the solution to the _Many Hot Encoder_ problem.  I'm dubbing this as _Many Hot Encoder_ because I don't know what else to call this problem.  It may or may not be common, but it is something that I have encountered more than once in my work and I think it is worth sharing with others.  Before getting into Many Hot Encoding, we'll quickly cover _One Hot Encoding_

## One Hot Encoding

I image most people reading this post have some experience with One Hot Encoding of Categorical variables, but for those who aren't familiar, One Hot Encoding (or OHE), is a way transform / encode categorical so that it can be included in machine learning models.  


As an example, suppose we have a dataset of subjects, along with a city in which the subject lives.  The field, which we will call `city` is a categorical variable and we are interested in seeing if the city in which the subject lives can used to explain the variation in the subject's income.  The defacto encoding of the `city` field would be a One Hot Encoding.  Given _n_ unique cities in the dataset, the One Hot Encoding will create _n_ new fields, one for each city, then when the subject is recorded to live in a city, the corresponding city column is set 1, while all other columns are set to 0.  

|id|	city| boston | nyc | tokyo |
|--|--|--|--|--|
|1|	boston|1|0|0|
|2	|nyc|0|1|0|
|3	|tokyo|0|0|1|
|4	|boston|1|0|0|
|5	|tokyo|0|0|1|
|6	|tokyo|0|0|1|
|..|	..|..|..|..|

## Many Hot Encoder

I don't know if it is appropriate to say that the _Many Hot Encoder_ is an extension of the _One Hot Encoder_, but I think it is easily understood as an extension.  Suppose you have a dataset where each record is a subject id, the subject id is not necessarily unique, meaning, it may appear more than once.  This would be typical in a transaction dataset, where records represent a transaction, or a line item from a transaction.  Items / features caught in a transaction

1. subject id
1. item name / sku / product number
2. cost of item

Another example of such a dataset could be a dataset of users and movies they have seen in the past year - and suppose you want to build a model to use past movie views to predict a future movie view.  
From here there would be a wealth of information.  





<!-- [__One Hot Encoding__](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) Encode categorical integer features using a one-hot aka one-of-K scheme.
The input to this transformer should be a matrix of integers, denoting the values taken on by categorical (discrete) features. The output will be a sparse matrix where each column corresponds to one possible value of one feature. It is assumed that input features take on values in the range [0, n_values).
This encoding is needed for feeding categorical data to many scikit-learn estimators, notably linear models and SVMs with the standard kernels.
Note: a one-hot encoding of y labels should use a LabelBinarizer instead. -->

<!-- source /home/tim/.rvm/scripts/rvm -->


This is very useful, but there are different instances where a similar type of feature construct would be
useful for certain ML application.  Let's suppose that you have collected transaction data, three columns in total:
1. user id - user id associated with the transaction
2. movie - name of movie

{% highlight scala %}
def printHi(name: String) = {
  println(s"Hi, $name")
}
printHi("Tom")
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %}
