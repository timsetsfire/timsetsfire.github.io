---
layout: post
title:  "Neural Networks from Scrach Part 1"
date:   2018-08-20 22:05:00 -0400
categories: jekyll update
---


## Motivation

This post came about from a lecture I had during my Udacity Deep Learning Nano Degree.  
The purpose of the lecture was to show how to build a neural network framework from
scratch (in Python) that mimicked TensorFlow.  

I decided that it would be fun to do the same in Scala, and attempt to add more functionity, and ultimately, be able to train an generative adversarial neural network on the MNIST dataset.  

Over the next several posts, I'll discuss
* Choosing a linear algebra Library
* Common operations in ND4J.  
* Linear Regression and Stochastic Gradient Descent in ND4J
* Approach to the Neural Network Framework
* Activation Functions
* Cost functions
* Simple Neural Network on MNIST
* Generative Adversarial Networks on MNIST.  

## Choosing a Linear Algebra Library   

The focus of this first post will be around which Linear Algebra library to use.  While there are many, I only considered between Breeze and ND4J / ND4S.  

### Considerations

Thinking about neural networks and what is needed to be accomplished with the data meant focusing  on how to carry out basic linear algebra operations as well manipulation of n-dimensional arrays.  This includes the following:

* shuffling of data along an axis
* slicing data along an axis.  
* matrix multiplies and addition
* elementwise operations

### Breeze

Prior to this exercise, my weapon of choice was [Breeze](https://github.com/scalanlp/breeze).  I was originally introduced to it when I was going through Pascal Bungion's book [Scala for Data Science](https://pascalbugnion.net/book.html) and given my experience with Matlab and Numpy I found the syntax very familiar and comfortable.  

Starting with breeze, I began to think through how I would accomplish the things listed above.  Basic matrix and elementwise operations were a easy breeze-y lemon squeezy, but slicing and shuffling where difficult difficult lemon difficult.  

The main shuffle operation I found in breeze was located in `breeze.linalg.shuffle`.  This implements the Fisher-Yates shuffle and based on
playing around with it, it will either shuffle the entire matrix, the rows, or the columns, but when shuffling, it will do the rows or columns independently.  

In Breeze, you can mimic Numpy operations on axes with the following syntax
`x(::, *)` will operate on axis 1, while x(*,::) will operate on axis 0.  

For example, consider below, where I want to shuffle the order of the rows, but you'll notice that the each column is shuffle indepenet of the others.  

{% highlight scala %}
scala> val x = new DenseMatrix(3, 2, (1 to 6).toArray )
x: breeze.linalg.DenseMatrix[Int] =
1  2
3  4
5  6

scala> breeze.linalg.shuffle(x(::, *) )  
res0: breeze.linalg.DenseMatrix[Int] =
3  6
5  2
1  4
{% endhighlight %}

Shuffling can also be accomplished with anything that is like `Seq[Int]`.  But, the issue \(maybe non-issue\\) with this is that the return is not a `DenseMatrix`, but a `breeze.linalg.SlicedMatrix[Int, Int, A]` where `A <: AnyVal`, and if you attempt a Matrix multiply, it returns the general `breeze.linalg.Matrix[A]`.  I don't think this is too big an issue, but may introduce typing problems when writing the frame work.  I should say it will introduce issues, since I'm not that great with Scala!  

Concerning slicing in Breeze, you can slice using `breeze.linalg.BitVector`, or anything we mentioned above for shuffling data.  Also available is `scala.collection.immutable.Range`.  This is useful when you can specify the indices you want in the  following manner `0 to n by k` or `0 until n by k`.  The `BitVector` is a Vector of Boolean values, essentially allowing you to turn rows on or off.  Please keep in mind this is what I have found.  If there are more / better ways to perform slicing, please let me know.  Using `Range` the return is `DenseMatrix`.  Using antyhing else, afaik, the return appears to be `SlicedMatrix[Int, Int, A]`, where `A <: AnyVal`.  

{% highlight scala %}
import breeze.linalg.{BitVector, DenseMatrix}
scala> val bv = BitVector(true, false, false)
bv: breeze.linalg.BitVector = BitVector(0)

scala> val x = new DenseMatrix(3, 2, (1 to 6).toArray )
x: breeze.linalg.DenseMatrix[Int] =
1  2
3  4
5  6

scala> x(bv, ::)
res0: breeze.linalg.SliceMatrix[Int,Int,Int] = 1  2
{% endhighlight %}

So far, not so bad, but I was hoping for something that would always return something consistent in terms of the data type.  

I think another reason at the time was storage of Breeze matrices.  I think loading large matrices proved problematic and I would run out of heap space - this is no issue in ND4J since their ndarrays are stored off heap.  

### ND4J

I came across ND4J at some point and have been interested in it ever since.  Its aim is to shorten the gap between JVM languages and Numpy or Matlab.  I would
encourage one to check out the this nice [numpy cheatsheet for nd4j](https://github.com/deeplearning4j/dl4j-examples/blob/master/nd4j-examples/src/main/java/org/nd4j/examples/numpy_cheatsheat/NumpyCheatSheat.java)

The main class that represents n-dimensional arrays in at `org.nd4j.linalg.api.ndarray.INDArray`.  You can usually operate on INDArrays in place or such that the return a new INDArray.  

Shuffling data is easy.  Suppose `x` is of type `INDArray` and has dimension $$n \times k$$, shuffling can be accomplished via

Below is the full REPL output
{% highlight scala %}

scala> import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

scala> val x = Nd4j.create( (1 to 6).toArray.map{_.toDouble}, Array(3,2), 'c')
x: org.nd4j.linalg.api.ndarray.INDArray =
[[1.00, 2.00],
 [3.00, 4.00],
 [5.00, 6.00]]

scala> Nd4j.shuffle(x, 0) // to shuffle the columns

scala> println(x)
[[2.00, 1.00],
 [4.00, 3.00],
 [6.00, 5.00]]

scala> Nd4j.shuffle(x, 1) // to shuffle the rows

scala> println(x)
[[2.00, 1.00],
 [6.00, 5.00],
 [4.00, 3.00]]

{% endhighlight %}

It is important to note that this shuffling occurs in place!  

Slicing is simple as well

{% highlight scala %}
scala> val rowsToGet = List(0,2)
rowsToGet: List[Int] = List(0, 2)

scala> val xSubSet = x.getRows( rowsToGet:_*)
xSubSet: org.nd4j.linalg.api.ndarray.INDArray =
[[2.00, 1.00],
 [4.00, 3.00]]

{% endhighlight %}

One thing to be away off - some operations that return an `INDArray` are returning references to
the original `INDArray`, for example, the matrix transpose.  For examples

{% highlight scala %}
scala> val xt = x.transpose
xt: org.nd4j.linalg.api.ndarray.INDArray =
[[1.00, 3.00, 5.00],
 [2.00, 4.00, 6.00]]

scala> x(0,1) = 10d
res67: org.nd4j.linalg.api.ndarray.INDArray =
[[1.00, 10.00],
 [3.00, 4.00],
 [5.00, 6.00]]

scala> xt
res68: org.nd4j.linalg.api.ndarray.INDArray =
[[1.00, 3.00, 5.00],
 [10.00, 4.00, 6.00]]
{% endhighlight %}

Aside from the considerations, I have considered also the fact that learning a new library would be particularly useful.  So, even if someone reading this points out embarrassingly simple solutions to what I'm moaning about above, I'd still stick by my choice of ND4J - just to learning something new.  

One other thing that may be worth pointing out.  ND4J can handle tensors, while I haven't figured out if Breeze can.  Tensors in this context are generalization of matrices to higher dimensions.  A good example would be an image for which we have 3 matrices of 28 by 28, with each matrix representing the R, G, and B values for the image.  We would have a tensor of shape (1, 3, 28, 28)
For example

{% highlight scala %}
import org.nd4j.linalg.factory.Nd4j

scala> val tensor = Nd4j.create( Array.fill{8}{util.Random.nextGaussian}, Array(2,2,2), 'c')
tensor: org.nd4j.linalg.api.ndarray.INDArray =
[[[-0.68, -2.19],
  [-0.15, -1.30]],

 [[1.76, -2.41],
  [-0.74, -2.25]]]

{% endhighlight %}

In the next posts, I'll likely focus on ND4J and some operations and processing snippets.  Hope
you enjoyed!
