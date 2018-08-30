---
layout: post
title:  "Neural Networks from Scrach Part 2"
date:   2018-08-20 22:05:00 -0400
categories: jekyll update
math: true
---



## Implementing Lasso with ND4J

First off - many thanks to [deep_thesis](http://deeplearningthesis.com/jekyll/mathematics/programming/2018/01/14/setting-up-jekyll.html) for help on setting up Jekyll for math!

This post, second in the neural nets from scratch entries, will focus on using ND4J / ND4S
to implement Lasso for linear regression (from scratch)  

This will get us comfortable with the way we use ndarrays in Scala and operations that will
be used throughout our neural network framework \( and then some! \).  

## Lasso

Least Absolute Shrinkage and Selection Operator, or Lasso for short, is a very popular method of feature selection in generalized linear models.  

We'll consider the Lasso problem as

$$argmin_{b} \|y - xb\|_2 + \lambda \|b\|_1$$

Here $$y$$ is a vector of continuous labels, $$x$$ is a matrix of features, $$b$$ is a vector
of weights and $$\lambda$$ is a hyperparameter used to control the sparsity of the regression solution.  For $$\lambda = 0$$ we have the standard solution and as $$\lambda$$ approaches $$\max |x^Ty| / n$$, $$b$$ goes to $$0$$.  

One cannot rely on regular gradient methods of optimization given this function is not differentiable on its domain, thus we'll use coordinate descent to solve.
 

## Gettin starting

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

From here, type `console` and you will enter REPL - Read, Evaluate, Print, Loop.  Once we get into the REPL, we run a few imports and create a few functions that will be used in coordinated descent.

{% highlight scala %}
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import scala.math.{abs, exp}

def signum(x: Double) = if(x==0) 0 else x / abs(x)

def softThresh(b: Double, gamma: Double) = {
  val sig = signum(b);
  val f = abs(b) - gamma  ;
  val pos = ( abs(f) + f ) /2;
  sig * pos;
}

def costFunction(x: INDArray, y: INDArray)(b: INDArray) = {
  val Array(m,n) = x.shape
  val error = (y - x.mmul(b))
  error.norm2Number.doubleValue / m
}

// generate some data
val x = Nd4j.randn(500,13)
val y = Nd4j.randn(500,1)

// initialize weights and bias
val Array(m,n) = x.shape
val w = Nd4j.zeros(n, 1)
val b = Array(0d).toNDArray

// standardize the data
// the arguments in the mean and std method denote the actual to calc along

val xMean = x.mean(0)
val xStd = x.std(0)
val yMean = y.mean(0)
val yStd = y.std(0)
val xs = x.subRowVector(xMean).divRowVector(xStd)
val ys = y.subRowVector(yMean).divRowVector(yStd)

val cost = costFunction(xs, ys)_

// Calculate the weights via normal equations
import org.nd4j.linalg.inverse.InvertMatrix.invert
val xTx = xs.transpose mmul xs
val what = invert(xTx,false) mmul xs.transpose mmul ys
println(what)

// run coordinate descent
val i = 0
val alpha = 1d
val lambda = 0.01
for(j <- 0 until 100) {
  for(i <- 0 until n) {
    val error = ys - xs.mmul(w)
    val update = xs(->, i).transpose.mmul( error + xs(->,i).mul(w(i,0))) / m
    w(i, 0) = softThresh(update(0,0) ,  alpha * lambda) / ( 1 + (1 - alpha)*lambda)
  }
  println(s"cost => ${cost(w)}")
}

{% endhighlight %}
