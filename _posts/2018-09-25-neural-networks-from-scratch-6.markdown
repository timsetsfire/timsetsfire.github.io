---
layout: post
title:  "Activation Functions"
date:   2018-09-13 22:05:00 -0400
categories: neural-nets
series: neural-nets
math: true
meta: Review some standard activation functions used in framework.
---

Activation functions are integral in the development of our neural network framework.  Recall that in the standard neural network, we have a set of inputs, of which we take linear combinations, and run those linear combinations through and activation function.  The output of the activation function then becomes the input for the next layer in the neural network.  

Some popular activation functions
* Identity
* Sigmoid
* Hypberbolic Tangent
* Maxout
* Rectified Linear Unit (ReLU)
* Noisy Rectified Linear Unit

All but the first activation function are nonlinear.  For our purpose, linear means $f(x + y) = f(x) + f(y)$ and $f(\alpha x) = \alpha f(x)$.  When a neural network uses identity activation functions for all layers, it should be clear this is equivalent to a single layer neural network work.  Why is nonlinearity important for neural networks?  It can afford a two layer neural network a universal function approximiation property - that is on closed and bounded subsets of $\mathbb{R}^n$, a two layer neural network can approximate any continuous functions with arbitrary precision.  

During the implementation of these activations functions, we will need a forward and a backward method available, that is, we need to know how to feed forward values and back propagate gradients.  

The Activation function model will extend the base `Node` class.  Below we'll walk through the sigmoid activation function.  The `forward` method should be obvious.  We'll actually just import the `sigmoid` function from `ND4J`, for backward, we must take a little bit of care.  If you recall, the `Node` class has an `inboundNodes` and `outboundNodes`, moreover, there is a gradients field that will help us manage the back propogation of gradients through the network.  I've included the `gradients` field in the `Sigmoid` class as a convenience.  The `gradients` field is map which has `Node` as key and `INDArray` as value.  The `INDArray` will be the gradient of the `Node`

### The constructor
{% highlight scala %}
class Sigmoid(node: Node) extends Node(node)
{% endhighlight %}
The constructor takes a single node.  For the purpose of a neural network, that node would probably be a `Linear` node - see the [Approach post]({{ site.baseurl }}{% post_url 2018-08-31-neural-networks-from-scratch-4 %})   

### The forward method

The forward method is the easiet method to implement.  The Activation has a field called `inboundNodes` since it extends `Node`.  The first element of `inboundNodes` is the node we passed to the constructor.  

{% highlight scala %}
Class Sigmoid(node: Node) extends Node(node) {

  override def forward(value: INDArray = null): Unit = {
    val in = inboundNodes(0)
    this.value = sigmoid(in.value)
  }

}
{% endhighlight %}
Easy enough!  One thing, notice that we do allow for a value to be passed in the forward method but we do nothing with it.  We could easily handle that so we could either pass a value or just feed the value from the inbound node, but I didn't do it here out of laziness.  

### The backward method

Backward is tough.  We Need to backprop all the gradients from the outbound nodes of our current node.  In order to accomplish this, we'll use some convenience fields to help with the book keeping.  

{% highlight scala %}
import scala.collection.mutable.{ArrayBuffer, Map=>MutMap}
import org.nd4j.linalg.ops.transforms.Transforms.sigmoid

class Sigmoid(node: Node) extends Node(node) {

  val gradients: MutMap[Node, INDArray] = MutMap()

  override def forward(value: INDArray = null): Unit = {
    val in = inboundNodes(0)
    this.value = sigmoid(in.value)
  }

  override def backward(value: INDArray = null): Unit = {
    // 1. we need to backprop to all the inbound nodes
    //    this will be done by using the gradients MutMap
    //    adding an instance to the map for each inbound node
    //    and initialize it to zeros matrix shaped like the
    //    value of this.value
    this.inboundNodes.foreach{
      n =>
        val Array(rows, cols) = this.value.shape
        this.gradients(n) = Nd4j.zeros(rows, cols)
    }
    // 2.  next we handle whether we manually pass back a gradient
    //     or backprop it from outbound nodes.  If we backprop, for each
    //     outbound node, we'll take its gradients map, and find the value
    //     for this node.  we'll calc the derivative of the activation
    //     function then take the hadamard (elementwise) product of that with
    //     gradient we are backproping
    if(value == null) {
      this.outboundNodes.foreach{
        n =>
        val gradCost = n.gradients(this)
        val sigmoid = this.value
        this.gradients(this.inboundNodes(0)) +=  sigmoid * (sigmoid.mul(-1d) + 1d) * gradCost
      }
    } else {
    // 3.  if we want to manually pass the gradient in there is much
    //     less to do - we don't loop through the outbound nodes
      this.gradients(this) = value
      val gradCost = this.gradients(this)
      val sigmoid = this.value
      this.gradients(this.inboundNodes(0)) +=  sigmoid * (sigmoid.mul(-1d) + 1d) * gradCost
    }
  }
}
{% endhighlight %}

## Companion Object

A companion object will be very useful for activation functions in general.  We can use the companion objects in a way such that we don't have to create the `Variable` nodes for all the linear combinations that we'll need.  

{% highlight scala %}
object Sigmoid {
  def apply(node: Node) = new Sigmoid(node)
  def apply(node: Node, size: (Any, Any)) = {
    val l1 = Linear(node, size)
    new Sigmoid(l1)
  }
}
{% endhighlight %}

Suppose that our dataset has 10 features, the companion object will allow use to do

{% highlight scala %}
val x = Input()
val s1 = Sigmoid(x, (15, 10))
val yhat = Sigmoid(s1, (10, 1))
{% endhighlight %}

as opposed to

{% highlight scala %}
val x = Input()
val z1 = Linear(x, (15,10))
val s1 = new Sigmoid(z1)
val z2 = Linear(s1, (10, 1))
val yhat = new Sigmoid(z2)
{% endhighlight %}
Yuck!




{% include series.html %}
