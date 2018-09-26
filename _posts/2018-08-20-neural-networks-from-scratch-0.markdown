---
layout: post
title:  "Neural Networks from Scrach"
date:   2018-08-20 22:05:00 -0400
categories: neural networks from scratch in scala
series: neural_networks
---


## Neural Networks from Scratch

Neural networks are everywhere!  There are very interesting and super powerful, and it seems you can not go anywhere on the web without finding a blog on neural networks from scratch.  So that the begs the following question - why even bother with another from scratch blog about neural networks.  

This blog sets itself apart, as far as I know, based on the language of choice and the implementation of neural nets.  


## Language

I think every post I have ever come across is in Python, not that there is anything wrong with that!  While Python may be lingua franca for data science, it is not the only one available.  The exposition here is done through Scala.  Scala is a fairly popular JVM language, and in my opinion offers a fairly robust alternative to Python for data science and machine learning problems with a few, potentially huge, exceptions - namely plotting.  

A few examples of the differences between Python and Scala

{% highlight python %}
>>> x = range(0, 3)
>>> for i in map(lambda j: j *2, x):
...   print(i,)
...
0
2
4
{% endhighlight %}

against is Scala equivalent
{% highlight scala %}
scala> val x = 0 until 3
x: scala.collection.immutable.Range = Range(0, 1, 2)

scala> x.map{ i => i * 2 }.foreach(println)
0
2
4
{% endhighlight %}

Or consider creates classes in Python vs scala

{% highlight python %}
>>> class Cat(object):
...   def __init__(self, name):
...     self.name = name
...   def __repr__(self):
...     return "Cat({})".format(self.name)
...
>>> c = Cat("steve")
>>> print(c)
Cat(steve)
{% endhighlight %}

and in Scala

{% highlight scala %}
scala> case class Cat(name: String)
defined class Cat

scala> val c = Cat("steve")
c: Cat = Cat(steve)

scala> println(c)
Cat(steve)
{% endhighlight %}

## Implementation

I don't think I have seen much in the way of implementations that offered a very flexible approach to neural networks.  The networks were typically coded out with 1 input layer, 2 hidden layers, 1 output and one cost function.  The derivatives were calc according to this very specific archecture, and if you wanted to add in another layer, it means writing additional code to calc derivatives.  

In this implementation we will use computational graphs and automatic differenation to aid in representing and training out neural networks.  This implementation was inspired (or copied) from a lecture I had during my nanodegree in Deep Learning with Udacity, the purpose of which was to show how to build a neural network framework from scratch (in Python) that mimicked TensorFlow - I have references at [scala-miniflow](https://github.com/timsetsfire/scala-miniflow).  

This framework will be flexible enough to handle continuous and categorical variables, several layers, various activation functions and not so straightforward architecures like generative adversarial networks.

It will not be able to handle Recurrent Networks or Convolutional Networks.  Those are #TODO


Over the next several posts, I'll discuss

{% include series.html %}
