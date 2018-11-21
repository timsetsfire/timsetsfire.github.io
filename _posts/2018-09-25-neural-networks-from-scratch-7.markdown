---
layout: post
title:  "Cost functions"
date:   2018-09-16 22:05:00 -0400
categories: neural-nets
series: neural-nets
math: true
meta: How to implement some standard cost functions.
---

The only way a neural network can be trained is if there is some objective.  Here we'll show a common cost function and it's implementation.  Since the cost function will fit into our framework as another node, it will need a `forward` method and a `backward` method, so revist the [Activation function post]({{ site.baseurl }}{% post_url 2018-09-25-neural-networks-from-scratch-6 %}) if you need a refresher on how those work.  

We'll consider the Binary Cross Entropy cost function, which is associated with neural networks used for binary classification.  

For binary classification, the target for our problem can take on one of two values, without loss of generality, we'll suppose those two values are 0 and 1.  Now, the output of our neural network will be a number between 0 and 1, and remember, this output is associated with the feed forward of the feature set through the neural network, and we'll say that the closer our output is to one, the more confident that the corresponding label (associated with the feature record) is 1, and the closer it is to 0, the more confident the assoicated label is 0.  

We can sum this up very concisely using the following formula

$$ y \log p + (1 - y) \log(1 - p)$$

where $y$ is the label for a record and $p$ is the result of feeding a record through our neural network.  The closer this value is to 0 means that the neural network is correctly labeling the dataset (spits out numbers closer to 0 when the actual label is 0 and numbers closer to 1 when the actual label is 1).  The above is Shannon Entropy (or Binary Cross Entropy or Bernoulli loglikelihood).  

Below we'll show the implementation of the Binary Cross Entropy, but in the actually framework, it will be implmented differently to ensure stability

{% highlight scala %}

class BCE(y: Node, yhat: Node) extends Node(y,yhat) {

  var diff = null.asInstanceOf[INDArray]

  override def forward(value: INDArray = null): Unit = {
    val y = this.inboundNodes(0).value
    val yhat = this.inboundNodes(1).value
    val obs = y.shape.apply(0).toDouble
    this.diff = (y / yhat) + ( y.mul(-1) + 1d) / (yhat.mul(-1) + 1d)
    val temp = ((y * log(yhat))) + ((y.mul(-1) + 1d)*log(yhat.mul(-1)+1d))
    this.value = temp.sum(0).div(obs.toDouble).mul(-1)
  }

  override def backward(value: INDArray = null): Unit = {
    val y = this.inboundNodes(0).value
    val yhat = this.inboundNodes(1).value
    val obs = y.shape.apply(0).toDouble
    this.gradients(this.inboundNodes(0)) = (log(yhat) - log( yhat.sub(1).mul(-1))).div(-obs)
    this.gradients(this.inboundNodes(1)) = ((y - yhat) / (yhat - yhat*yhat)).div(-obs)

  }
}
{% endhighlight %}

{% include series.html %}
