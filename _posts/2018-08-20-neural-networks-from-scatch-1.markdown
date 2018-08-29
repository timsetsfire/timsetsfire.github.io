---
layout: post
title:  "Neural Networks from Scrach Part 1"
date:   2018-08-20 22:05:00 -0400
categories: jekyll update
---


## Motivation

This post came about from a lecture I had during my Udacity Deep Learning Nano Degree.  
The purpose of the lecture was to show how to build a neural network framework from
scratch \(in python\\) that mimicked TensorFlow.  

I decided that it would be fun to do the same in Scala, and attempt to add more functionity, by adding several cost and activation functions for use in training a neural network.

Over the next several posts, I'll discuss
* Choosing a linear algebra Library
* Approach to the Framework
* Cost functions
* Activation Functions
* Simple Neural Network on MNIST
* Generative Adversarial Networks on MNIST --- hopefully.  

## Choosing a Linear Algebra Library   

### Considerations

Thinking about neural networks and what is needed to be accomplished with the data meant focusing  on
how to carry out basic linear algebra ops as well manipulation of n-dimensional arrays.  This includes the following:

* shuffling of data along an axis
* slicing data along an axis.  
* matrix multiplies and addition
* element wise operations

### Breeze

Prior to this exercise, my weapon of choice was [Breeze](https://github.com/scalanlp/breeze).  I was originally introduced to
it when I was going through Pascal Bungion's book [Scala for Data Science](https://pascalbugnion.net/book.html) and given my experience with Matlab and Numpy I found the syntax very familiar and comfortable.  

Started with breeze, I started to think through how I would accomplish the things listed above.  Basic matrix and elementwise operations were a easy breeze-y lemon squeezy, but slicing and shuffling where difficult difficult lemon difficult.  

The main shuffle operation I found in breeze was located in `breeze.linalg.shuffle`.  This implements the Fisher-Yates shuffle and based on
playing around with it, it will either shuffle the entire matrix, the rows, or the columns, but when shuffling, it will do the rows or columns
independently.  

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

 In Breeze, you can mimic Numpy operations on axes with the following syntax
`x(::, *)` will operate on axis 1, while x(*,::) will operate on axis 0.  

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

Aside from the considerations, I have considered also the fact that leanrning a new library would be
particularly useful.  SO even if someone reading this points out embarrassingly simple solutions
to what I'm moaning about above, I'd still stick by my choice of ND4J - just to learning something new.  

One other thing that may be worth pointing out.  ND4J can handle tensors, while I haven't figured out if Breeze can.  

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

<!--


### In breeze

Shuffling in breeze is not a breeze.  I never found a simple way to do this.  `breeze.linalg.shuffle` shuffles columns and rows independently,
which made things difficult.  I found that it will shuffle the rows / columns independently.  
For example suppose that you have a $$n \time k$$ matrix of data and it is stored in the variable `x`.


Simply enough, but breeze will not do the shuffling in place.  

*Slicing vectors and rows with non continguous indices.  This can be accomplished with `breeze.linalg.BitVector`, but I found this to be quite
cumbersome.




change## Neural Networks



$$ x^2 + y^2 = z^2 $$

Inline math $$ e^2 $$ for inline equations!!

$$ e^2 $$

Testing some math input

but with text around it $$ e^{i\pi} + 1 = 0 $$



## Framework

{% highlight scala %}
trait Node
{% endhighlight %}

bundle exec jekyll serve ~
