---
layout: post
title:  "Neural Networks from Scrach Part 3"
date:   2018-08-29 22:05:00 -0400
categories: jekyll update
math: true
---



## Implementing Linear Regression with ND4J

First off - many thanks to [deep_thesis](http://deeplearningthesis.com/jekyll/mathematics/programming/2018/01/14/setting-up-jekyll.html) for help on setting up Jekyll for math!

Within this post, will focus on using ND4J / ND4S solve a linear regression problem via normal equations, and implement mini-batch Stochastic Gradient Descent to estimate a regression model!

This will get us comfortable with the way we use ndarrays in Scala and operations that will
be used throughout our neural network framework, while introducing a popular method to learn neural nets.  


## Regression

I'll assume that anyone reading this is familiar with linear regression, so I'll keep those details spares.  Let's jump directly into estimate linear regression with normal equations in ND4J.  

In the root of your project folder, create `code/regression.scala`

{% highlight scala %}

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.nd4j.linalg.ops.transforms.Transforms.{pow,normalizeZeroMeanAndUnitVariance=>stdize}
import org.nd4j.linalg.inverse.InvertMatrix.invert

def r2(y: INDArray, yhat: INDArray) = {
  val rss = pow(y sub yhat,2)
  val tss = pow(y.subRowVector(y.mean(0)),2)
  1d - rss.sumT / tss.sumT
}

// read in data
val x = Nd4j.readNumpy("resources/boston_x.csv", ",")
val y = Nd4j.readNumpy("resources/boston_y.csv", ",")

// standardize data.  
// using the imported stdize will standardize inplace
val xs = x.subRowVector(x.mean(0)).divRowVector(x.std(0))
val ys = y.subRowVector(y.mean(0)).divRowVector(y.std(0))

// estimate weight matrix b
val b = invert(xs.transpose mmul xs, false) mmul xs.transpose mmul ys

val yhat = xs mmul b

println( f"r^2: ${r2(ys,yhat)}%2.3f")

{% endhighlight %}

Now, open terminal and run `sbt console`, and once you are in REPL, run `:load code/regression.scala`

## Stochastic Gradient Descent

If you aren't familiar with gradient descent, check out wikipedia.  "Stochastic gradient descent (SGD) and its variants are probably the most used optimization algorthsm for machine learning in general and for deep learning in particular" - Deep Learning.  

The algorithm is as follows

{% highlight scala %}
while stopping criterion no met do:
  sample a minibatch of m examples from the training set
  computer gradient estimate: g
  update parameters: b -= learning_rate * g
{% endhighlight %}

Sampling could happen in such a way that the algorithm converges before it every sees all the training examples.  

{% highlight scala %}


import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.nd4j.linalg.ops.transforms.Transforms.{pow,normalizeZeroMeanAndUnitVariance=>stdize}
import org.nd4j.linalg.inverse.InvertMatrix.invert

// read in data
val x = Nd4j.readNumpy("resources/boston_x.csv", ",")
val y = Nd4j.readNumpy("resources/boston_y.csv", ",")

def r2(y: INDArray, yhat: INDArray) = {
  val rss = pow(y sub yhat,2)
  val tss = pow(y.subRowVector(y.mean(0)),2)
  1d - rss.sumT / tss.sumT
}

def cost(y: INDArray, x: INDArray, b: INDArray) = {
 (y - x.mmul(b)).norm2T / y.shape.apply(0).toDouble
}

// standardize data.  
// using the imported stdize will standardize inplace
val xs = x.subRowVector(x.mean(0)).divRowVector(x.std(0))
val ys = y.subRowVector(y.mean(0)).divRowVector(y.std(0))

// initialize weights
val b = Nd4j.randn(13,1)

val Array(xrows, xcols) = xs.shape
val batchSize = 128
val stepsPerEpoch = xrows / batchSize
val epochs = 500
val t = new java.util.concurrent.atomic.AtomicInteger
val data = Nd4j.concat(1, ys, xs)
val learningRate = 0.01

for(epoch <- 0 to epochs) {

  var loss = 0d
  var costValue = 0d
  var n = 0d

  for(steps <- 0 to stepsPerEpoch) {

    t.addAndGet(1)
    Nd4j.shuffle(data,1)
    val xBatch = data( 0 until batchSize, 1 until 14)
    val yBatch = data( 0 until batchSize, 0)
    val yhat = xBatch.mmul(b)
    val grad = xBatch.transpose.mmul(yBatch sub yhat).div(batchSize).mul(-1)
    b.subi(grad.mul(learningRate))
    loss += cost(yBatch, xBatch, b) * xBatch.shape.apply(0)
    n += xBatch.shape.apply(0)

  }

  if(epoch % 10 == 0) println(f"cost: ${loss/n}%2.3f")

}
{% endhighlight %}
