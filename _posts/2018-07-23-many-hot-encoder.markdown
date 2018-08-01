---
layout: post
title:  "Many Hot Encoder"
date:   2018-07-23 23:05:00 -0400
categories: jekyll update
---

source /home/tim/.rvm/scripts/rvm

This post explores the solution to the _Many Hot Encoder_ problem.  I'm dubbing this as _Many Hot Encoder_ because I don't know what else to call this problem.  It may or may not be common, but it is something that I have encountered more than once in my work and I think it is worth sharing with others.  Before getting into Many Hot Encoding, we'll quickly cover _One Hot Encoding_

## One Hot Encoding

I imagine most people reading this post have some experience with One Hot Encoding of Categorical variables, but for those who aren't familiar, One Hot Encoding (or OHE), is a way transform / encode categorical so that it can be included in machine learning models.


As an example, suppose we have a dataset of subjects, along with a city in which each subject lives.  The field containing the city, which we will call `city`, is a categorical variable and we are interested in seeing if the city in which the subject lives can used to explain the variation in the subject's income.  The defacto encoding of the `city` field would be a One Hot Encoding.  Given _n_ unique cities in the dataset, the One Hot Encoding will create _n_ new fields, one for each city, then when the subject is recorded to live in a city, the corresponding city column is set 1, while all other columns are set to 0.  See below

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

Another example of such a dataset could be a dataset of users and movies the use has seen in the past year or two - and suppose you want to build a model to use past movie views to predict a future movie view.

Let's consider the example of the movie dataset.  Our dataset contains two fields, one for user id and another for the movie they have seen in the past year two.

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

user|	jumanji|	a quiet place|	game night|	jurasic world|	the last jedi|	hereditary|	coco|	mother|	..|	something else|
--  |--   |-- |-- |-- |-- |-- |-- |-- |-- |--|
1|	1|	1|	0|	0|	1|	0|	0|	0|	..|	0|
2|	0|	0|	1|	1|	0|	0|	1|	0|	..|	0|
3|	0|	0|	0|	0|	0|	0|	0|	1|	..|	0|
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
val data = src.getLines.map{_.split(",")}.toList.tail // drop the headers
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

The solution in spark is significantly easier.  A lot of smart people have put in a lot of hard work to make this solution very straight forward.

We'll assume that you already have a `SparkContext` available as `sc`.  Furthermore, we'll suppose that the dataset is in a Spark DataFrame called `df`.

The columns in `df` are `user_id` and `movie_name`.  We will not assume that the data is sorted in any way, and that there is the possibility of duplicate records, i.e., users can see a movie more than once.  I have a friend who saw _Firefly_ in the theatre 4 times!

Our plan of attack is as follows
1. Dedup the data.  We are only recording whether the user saw a movie, not how many times they saw it
2. Fit a `StringIndexer` on the `movies` column.
3. Transform the `movies` column to a new column `movies_si`.
4. run `collect_list` on `movies_si`.  This is used as the indices in the `SparseVector`.
5. Generate the `SparseVector`.

{% highlight scala %}
import org.apache.spark.ml.feature.StringIndexer
import org.apahce.spark.ml.linalg.SparseVector

val df = sc.parallelize(
  List(
    (1,"Jumani"),
    (1,"A Quiet Place"),
    (1,"Game Night"),
    (2,"Jurasic World"),
    (2,"The Last Jedi"),
    (2,"Hereditary"),
    (2,"Coco"),
    (3,"Mother")
  )
).toDF("userid", "movie_name")

val dfDistData = df.distinct
val si = new StringIndexer()
si.setInputCol("movie_name").setOutputCol("movie_name_si")
val si_fit = si.fit(dfDistData)
si_fit.transform(dfDistData).createOrReplaceTempView("movie_data")
val dfDistDataAgg = spark.sql(
  """select userid, collect_list(movie_name_si) as movies
     from movie_data
     group by 1"""
     )
{% endhighlight %}

We can get the movies and the associated index via
{% highlight scala %}
si_fit.labels.zipWithIndex.take(10).foreach(println)
{% endhighlight %}

```
(The Last Jedi,0)
(Jumani,1)
(Hereditary,2)
(Jurasic World,3)
(Mother,4)
(Game Night,5)
(Coco,6)
(A Quiet Place,7)
```

`dfDistDataAgg` DataFrame is shown below.

```
+------+--------------------+
|userid|              movies|
+------+--------------------+
|     1|     [1.0, 5.0, 7.0]|
|     3|               [4.0]|
|     2|[0.0, 3.0, 2.0, 6.0]|
+------+--------------------+
```
The plan is to use the arrays in the movies field as the Indices of the non-zero
elements of a `org.apache.spark.ml.linalg.SparseVector`, we'll also need to pass
through  `value` and `length` as we did for our for scratch solution.  For `value`,
we'll pass an array of ones of the same length as the `movies` array for the values,
and we'll use the `labels` method of the `StringIndexer` to set the `length` of the
 `SparseVector`.  But, before we do that, `org.apache.spark.ml.linalg.SparseVector` expects indices to be
sorted.

{% highlight scala %}
import org.apache.spark.sql.functions.udf
import scala.collection.mutable.WrappedArray
val toSparseVector = udf {
  in: WrappedArray[Double] =>
    val size = si_fit.labels.length
    val index = in.toArray.map{ _.toInt}.sortWith{ _ < _ }
    val values = in.toArray.map{ i => 1d}
    new SparseVector(size, index, values)
}
dfDistDataAgg.select(
  dfDistDataAgg.col("userid"),
  toSparseVector(dfDistDataAgg.col("movies"))as("mhe")
  )
{% endhighlight %}
and viola
```
+------+--------------------+
|userid|                 mhe|
+------+--------------------+
|     1|(8,[1,5,7],[1.0,1...|
|     3|       (8,[4],[1.0])|
|     2|(8,[0,2,3,6],[1.0...|
+------+--------------------+
```
