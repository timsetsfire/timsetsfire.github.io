---
layout: post
title:  "Neural Networks from Scrach Part 2"
date:   2018-08-20 22:05:00 -0400
categories: jekyll update
math: true
---

## Introduction to ND4J

First off - many thanks to [deep_thesis](http://deeplearningthesis.com/jekyll/mathematics/programming/2018/01/14/setting-up-jekyll.html) for help on setting up Jekyll for math!

This post, second in the neural nets from scratch entries, will focus on show how to perform some basic matrix operations in ND4J.  I'll likely list the comparable numpy syntax given its ubiquity.  

## Getting starting

The easiest way to get ND4J and the Scala bindings is via Sbt.  Download and install [Sbt](https://www.scala-sbt.org/download.html).  Create a folder which will house
the project.  Create a file called `build.sbt` and place the following lines in the files
{% highlight scala %}
name := "scala-miniflow"

version := "0.1.0"

scalaVersion := "2.11.8"

val nd4jVersion = "0.7.2"

libraryDependencies += "org.nd4j" % "nd4j-native-platform" % nd4jVersion

libraryDependencies += "org.nd4j" %% "nd4s" % nd4jVersion
{% endhighlight %}

Now, open terminal wherever the build file is located and run

`sbt`

Give this some time to build.  The Scala Build Tool is grabbing requirements.  Once completed you will see

```
sbt:scala-miniflow>
```

From here, type `console` and you will enter REPL - Read, Evaluate, Print, Loop.  Once we get into the REPL, we can get started.

## Creating a Matrix

{% highlight scala %}
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
val x1 = Nd4j.create( Array[Double](2,7,6,9,5,1,4,3,8), Array(3,3), 'c')
// or
val x2 = Array(Array(2,7,6),Array(9,5,1),Array(4,3,8)).toNDArray
{% endhighlight %}

Notice in the contruction of `x1`, we specify `'c'`.  This will be column major, while `'f'` would have resulted in row major.  

Output:
{% highlight scala %}
x1: org.nd4j.linalg.api.ndarray.INDArray =
[[2.00, 7.00, 6.00],
 [9.00, 5.00, 1.00],
 [4.00, 3.00, 8.00]]

x2: org.nd4j.linalg.api.ndarray.INDArray =
[[2.00, 7.00, 6.00],
 [9.00, 5.00, 1.00],
 [4.00, 3.00, 8.00]]
{% endhighlight %}

## Tranpose

Very fundamental operation is transposition - this is switching the row index with the column index for each element.

{% highlight scala %}
println(x1)
println(x1.transpose)
{% endhighlight %}

In REPL
{% highlight scala %}
scala> println(x1)
[[2.00, 9.00, 4.00],
 [7.00, 5.00, 3.00],
 [6.00, 1.00, 8.00]]

scala> println(x1.transpose)
[[2.00, 7.00, 6.00],
 [9.00, 5.00, 1.00],
 [4.00, 3.00, 8.00]]
{% endhighlight %}

One thing of particular importance to put out, in ND4J, `transpose` is one of several operations that return a reference to the original object, and not necessarily creating a new one.  That means if `x` is an $$n \times k$$ matrix, and you create `xt = x.transpose` and set `xt(0,1) = 10`, then `x(1,0)` will be set to 10 as well.  We'll see the other operations that act similarly when we discuss slicing.  


## Slicing

The fundamental methods of slicing in ND4J involve `get`, `getRow`, `getRows`, `getColumn`, and lastly, `getColumns`.  

`get`, `getRow`, `getColumn` all return objects that point to the same underlying data in memory, much as `transpose` does.  This can be very useful, but you must remain mindful.  

{% highlight scala %}
val rowsToGet = List(0,2)
xSubSet = x1.getRows( rowsToGet:_*)
{% endhighlight %}

Output:
{% highlight scala %}
xSubSet: org.nd4j.linalg.api.ndarray.INDArray =
[[2.00, 1.00],
 [4.00, 3.00]]
{% endhighlight %}

Changing an element of `xSubSet` will not affect the original matrix.  But do notice this though,

{% highlight scala %}
val row1 = x1.getRow(1)
println(row1)
row1(0,0) = 10
println(row1)
println(x1)
{% endhighlight %}

In REPL
{% highlight scala %}
scala> val row1 = x1.getRow(1)
row1: org.nd4j.linalg.api.ndarray.INDArray = [9.00, 5.00, 1.00]

scala> println(row1)
[9.00, 5.00, 1.00]

scala> row1(0,0) = 10
res35: org.nd4j.linalg.api.ndarray.INDArray = [10.00, 5.00, 1.00]

scala> println(row1)
[10.00, 5.00, 1.00]

scala> println(x1)
[[2.00, 7.00, 6.00],
 [10.00, 5.00, 1.00],
 [4.00, 3.00, 8.00]]
{% endhighlight %}

There are some useful implicits like `->`, and `n to m by k`

{% highlight scala %}
scala> val x = Nd4j.randn(3,6)
x: org.nd4j.linalg.api.ndarray.INDArray =
[[0.47, 1.15, -0.31, 0.30, 0.27, 0.02],
 [0.11, -0.57, 1.80, 1.21, -0.32, -0.07],
 [0.51, 0.62, 0.05, 0.80, -0.09, 0.29]]

scala> x(->, 0 to 6 by 3)
res39: org.nd4j.linalg.api.ndarray.INDArray =
[[0.47, 0.30],
 [0.11, 1.21],
 [0.51, 0.80]]
{% endhighlight %}

The `->` operator will take all, while `0 to 6 by 3` will take then 0 and 3 columns of the matrix.  This type of slicing will return a reference, so be careful if you act on the elements of the sub matrix, as you will change the original.  

### Slicing Tensors

As pointed out before, creating Tensors in `ND4J` is pretty simply.  
{% highlight scala %}
import org.nd4j.linalg.factory.Nd4j

scala> val tensor = Nd4j.create( Array.fill{8}{util.Random.nextGaussian}, Array(1,2,2,2), 'c')
tensor: org.nd4j.linalg.api.ndarray.INDArray =
[[[[-1.03, 1.15],
   [0.30, 0.84]],

  [[0.12, -0.18],
   [0.60, -0.47]]]]

{% endhighlight %}

Slicing this tensor is fairly intuitive with imported implicits.

{% highlight scala %}
import org.nd4s.Implicits._
scala> tensor(0,1,->,->)  // tensor(0,1,->) also works
res0: org.nd4j.linalg.api.ndarray.INDArray =
[[0.12, -0.18],
 [0.60, -0.47]]
{% endhighlight %}


## Elementwise operations

`INDArray` have elementwise methods availble.  Some standards, `add`,  `mul`, `sub`, `div`.  The argument must be the same size as the matrix or a scalar.  

All the operations listed above have an inplace version as well: `addi`, `muli`, `subi`, and `divi`.  For example `addi` would be equivalent to `x += 1`.  

More elementwise operations are available in `org.nd4j.linalg.ops.transforms.Transforms`, for example, `exp`, `log`, etc.  

## Broadcasting

I really like how ND4J handles broadcasting - it doesn't!  There are methods available to handle this, `addRowVector`, `mulRowVector`, `subRowVector`, and `divRowVector`.  These will be used a lot when we write our framework.  There are similar operations for column vectors.  Moreover, there are inplace versions available, for example `addiRowVector`.  

## Matrix Multiplication

Given two objects, `x` and `y`, of type `INDArray`, that have shapes which make them compatible for matrix mutliplication, we can matrix multiply via

{% highlight scala %}
x.mmul(y)
// or
x mmul y
{% endhighlight %}


## Summary statistics

Many statistics are available similar to numpy.  Suppose you have data in an array `x`.  You can get the column means via `x.mean(0)` and the row means with `x.means(1)`.  Other operations like sum, std, et cetera are available.  

## Loading Data

From csv - this is very straight forward.  

{% highlight scala %}
val x_ = Nd4j.readNumpy("resources/mnist_test.csv", ",").
{% endhighlight %}

## Other useful things

Numpy.ones and Numpy.zeros equivalent
{% highlight scala %}
val ones = Nd4j.ones(2,2)
val zeros = Nd4j.zeros(2,2)
{% endhighlight %}

In REPL

{% highlight scala %}
scala> val ones = Nd4j.ones(2,2)
ones: org.nd4j.linalg.api.ndarray.INDArray =
[[1.00, 1.00],
 [1.00, 1.00]]

scala> val zeros = Nd4j.zeros(2,2)
zeros: org.nd4j.linalg.api.ndarray.INDArray =
[[0.00, 0.00],
 [0.00, 0.00]]
 {% endhighlight %}

Numpy.concatenate equivalent
{% highlight scala %}
val onesAndZeros = Nd4j.hstack(ones, zeros)
val onesAndZeros2 = Nd4j.vstack(ones, zeros)
{% endhighlight %}

In REPL
{% highlight scala %}
scala> val onesAndZeros = Nd4j.hstack(ones, zeros)
onesAndZeros: org.nd4j.linalg.api.ndarray.INDArray =
[[1.00, 1.00, 0.00, 0.00],
 [1.00, 1.00, 0.00, 0.00]]

scala> val onesAndZeros2 = Nd4j.vstack(ones, zeros)
onesAndZeros2: org.nd4j.linalg.api.ndarray.INDArray =
[[1.00, 1.00],
 [1.00, 1.00],
 [0.00, 0.00],
 [0.00, 0.00]]
{% endhighlight %}

You can pass as many arrys to `vstack` and `hstack` provided they conform in shape.  
Similarly, you could use `Nd4j.concat(axis: Int, v: INDArrays*)` to accomplish the same, 0 will `hstack`, and 1 will `vstack`.  
