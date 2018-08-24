---
layout: post
title:  "Neural Networks from Scrach"
date:   2018-08-20 22:05:00 -0400
categories: jekyll update
math: math.html
---


{% include {{ page.math }} %}

### Neural Networks from Scratch

## Motivation

This post came about from a lecture I had during my Udacity Deep Learning Nano Degree.  
The purpose of the lecture was to show how to build a neural network framework from
scratch (in Python) that mimicked TensorFlow.  

I decided that it would be fun to do the same in Scala, and attempt to add more functionity, by adding
several cost and activation functions for use in training a neural network.

Over the next several posts, we'll cover
* Choosing a linear algebra Library
* Approach to the Framework
* Cost functions
* Activation Functions
* Simple Neural Network (on MNIST)
* Generative Adversarial Networks (on MNIST) --- hopefully.  

### Choosing a Linear Algebra Library.    

## Considerations

Thinking about neural networks and what is needed to be accomplished with the data required me to focus on
how to carry out linear algebra ops as well as data processing / transformation operations on n-dimensional arrays like the following:

* shuffling of data along an axis
* slicing data along an axis.  
* matrix multiplies
* element wise operations

### Breeze

Prior to this exercise, I used [Breeze](https://github.com/scalanlp/breeze) for everything.  I was originally introduced to
it when I was going through Pascal Bungion's Book [Scala for Data Science](https://pascalbugnion.net/book.html) and
just found it easy.  I had experience with Matlab and Numpy and found the syntax very familiar and comfortable.  

When I dove into this project, I went right to breeze and considered how to accomplish the things listed above.  Matrix multipies and
elementwise operations were a easy breeze-y lemon squeezy, but slicing and shuffling where difficult difficult lemon difficult.  

The main shuffle operation I found in breeze was located in `breeze.linalg.shuffle`.  This implements the Fisher-Yates shuffle and based on
playing around with it, it will either shuffle the entire matrix, the rows, or the columns, but when shuffling, it will do the rows or columns
independently.  

{% highlight %}
scala> x
res78: breeze.linalg.DenseMatrix[Int] =
1  2
3  4
5  6

scala> breeze.linalg.shuffle(x(::, \*) )
res79: breeze.linalg.DenseMatrix[Int] =
3  6
5  2
1  4
{% endhighlight %}

Or you can shuffle with anything that is like `Seq[Int]`.  But, the issue (maybe non-issue) with this is that the return is not a `DenseMatrix`, but a
`breeze.linalg.SlicedMatrix[Int, Int, A]` where `A <: AnyVal`, and if you attempt a Matrix multiply, it returns the general `breeze.linalg.Matrix[A]`.  
I don't think this is too big an issue, but may introduce typing problems when writing the frame work.  I should say it will introduce issues, since I'm
not that great with Scala!  

Concerning slicing, in Breeze, you can size using `breeze.linalg.BitVector`, or anything we mentioned above for shuffling data.  Also available is
`scala.collection.immutable.Range`.  This is useful when you can specify the indices you want in the  following manner `0 to n by k` or `0 until n by k`.  
The `BitVector` is a Vector of Boolean values, essentially allowing you to turn rows on or off.  Please keep in mind this is what I have found.  
If there are more / better ways to perform slicing, please let me know.  Using `Range` the return is `DenseMatrix`.  Using antyhing else, the return appears to
be `SlicedMatrix[Int, Int, A]`, where `A <: AnyVal`.  

{% highlight %}
scala> val bv = BitVector(true, false, false)
bv: breeze.linalg.BitVector = BitVector(0)

scala> x
res88: breeze.linalg.DenseMatrix[Int] =
1  2
3  4
5  6

scala> x(bv, ::)
res89: breeze.linalg.SliceMatrix[Int,Int,Int] = 1  2
{% endhighlight %}

So far, not so bad, but I was hoping for something that would always return something consistent in terms of the data type.  

### ND4J

I came across ND4J at some point and have always been interested in it.  Its aim is to shorten the gap between JVM languages and Numpy or Matlab.  I would
encourage one to check out the this nice [numpy cheatsheet for nd4j](https://github.com/deeplearning4j/dl4j-examples/blob/master/nd4j-examples/src/main/java/org/nd4j/examples/numpy_cheatsheat/NumpyCheatSheat.java)

Shuffling data is easy.  Suppose `x` is of type `org.nd4j.linalg.api.ndarray.INDArray` and has dimension $$n \times k$$

{% highlight scala %}
import org.nd4j.linalg.factory.Nd4j

Ndj4.shuffle(x, 0) // to shuffle the columns
Nd4j.shuffle(x, 1) // to shuffle the rows
{% endhighlight %}

It is important to note that this shuffling occurs in place!

Slicing is simple as well

{% highlight scala %}

val xSubSet = x.getRows( rowsToGet:\_\*)

{% endhighlight %}

This is not creating a new `INDArray` of the `rowsToGet`, but referencing them.  So any changes made inplace to `xSubSet` are
made to `x` - I'm pretty sure.  This way you always know the type you are dealing with.  There is enough information here to make the choice,
but I also considered that it would be useful to learn a new library.  SO even if someone reading this points out embarrassingly simple solutions
to what I'm moaning about above, I'd still stick by my choice of ND4J - just to learning something new.  

In the next posts, I'll likely focus on using ND4J.  

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

bundle exec jekyll serve ~ -->
