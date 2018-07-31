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

Let's consider the example of the movie dataset.  Our dataset contains two fields, one for user id and another for the movie they have seen in the past year (or whenever).  

User	|movie
--|--
1|	Jumani
1|	A Quiet Place
1|	Game Night
2|	Jurasic World
2|	The Last Jedi
2|	Hereditary
2|	Coco
3|	Mother
..|..

Our aim is to turn the above table into something like

user|	m1|	m2|	m3|	m4|	m5|	m6|	m7|	m8|	..|	mn|
--  |--   |-- |-- |-- |-- |-- |-- |-- |-- |--|
1|	1|	1|	0|	0|	1|	0|	0|	0|	..|	0|
2|	0|	0|	1|	1|	0|	0|	1|	1|	..|	0|
3|	0|	0|	0|	0|	0|	0|	0|	0|	..|	0|
..|..| ..| ..| ..| ..| ..| ..| ..| ..|.. |

so that we may use it in a machine learning algorithm.  

One thing that should be clear is that our date will considerably sparse, if there are around 700 movies released each [year](https://www.quora.com/On-average-how-many-Hollywood-films-are-released-in-a-year)
and if we have 2-3 years of data, our data could become fairly wide.  

## Considerations

At first glance it may seem necessary to have some information about the following
1.  How many unique movies are in the file
2.  Can a user-movie combination appear more than once?  
3.  Is the data sorted.  

Each of these could change how we would process our dataset, but for our purposes, this will not have any impact of the given solution.  

## From Scratch Solution

The idea for this is fairly straight forward, we'll create somethign that operates like a Spark StringIndexer, and we'll pivot our data so that the rows are user ids, and the columns are movies (index movies).  

Our from scratch solutuion will be done with all base modules available in Scala, and one Sparse Vector case class meant to efficiently store our movie data.  

{% highlight scala %}
case class SparseVector(length: Int, indices: Vector[Int], values: Vector[Double])
{% endhighlight %}

Notice that the constructor for the `SparseVector` takes `length`, `indices` - indices of non-zero values, and finally `values` of the non-zero elements.  

The movies will be stored in a `scala.collection.mutable.Map` with movie name as the key and an integer as the value.  This integer will serve as an index in the SparseVector, so that if this integer appears in the `indices` field, then the user has seen the movie.  The general idea, we'll initialize a counter to 0, and as well iterate over each record, if the movie is in the `Map`, then nothing, otherwise, we add the movie to the map as a key, and add the value of the counter as the value, then the counter is inremented by 1.  

We'll use `java.util.concurrent.atomic.AtomicInteger` as the counter.  

Lastly, we won't make any assumptions around the order of the records.  That is, a users movie listing may not be contiguious, which is no issue.  We'll get around this be creating a `Map` with key user id and value `Vector[Int]` of the movies the user has seen.  

{% highlight scala %}
// full code
import scala.collection.mutable.{Map=>MutMap}
import java.util.concurrent.atomic.AtomicInteger

case class SparseVector(length: Int, indices: Vector[Int], values: Vector[Double])

val src = scala.io.Source.fromFile("movies.csv")
val data = src.getLines.map{ _.split(",")}.toList.tail // drop the headers
val users = MutMap[String, Vector[Int]]()              // map of users
val movies = MutMap[String, Int]()                     // StringIndexer
val counter = new AtomicInteger                        // Incrementer for StringIndexer

data.foreach{ line =>
  val Array(user, movie) = line
  if(!movies.contains(movie)) {
      movies.update(movie, counter.getAndAdd(1))  
  }
  val movieId = movies(movie)
  if(users.contains(user)) {
      users.update(user, movieId +: users(user))
  } else {
      users.update(user, Vector(movieId))
  }
}
for(user <- users.keys) {
   users.update( user, users(user).sortWith(_ < _))
}
val ohe = users.map{
  case(user, movies) => (user, SparseVector(counter.get, movies, Vector.fill(movies.length){1d}))
}
{% endhighlight %}

## Using Spark



### What you should now
1. Reading files
2. loops
3. dictionary
4. storing sparse data
5. writing classes

Approach

* Do not assume the data is sorted.
* Assume the user movie combination is unique

We'll initialize a variable c to keep track of the number of unique movies in the data set, then convert our dataset to key value pairs where each key is a user id and each value is an array containing the movie id which the user has seen. These arrays will then be convert to a sparse vector, for us this will just be a tuple of type  (Int, Vector[Int], Vector[Double]). The first element in the tuple is the length of the vector, the second element will be the indices which are non-zero, while the third element will contain the corresponding non-zero values.




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
