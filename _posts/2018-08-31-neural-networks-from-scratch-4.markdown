---
layout: post
title:  "Neural Networks from Scrach Part 4"
date:   2018-09-01 22:05:00 -0400
categories: jekyll update
math: true
---

## Graphs

Still in progress - (as are all other neural net posts)

Our approach to neural networks in scala will be based on graphs.  A graph is what is meant by a network in the a neural network and is a structure made up of edges and vertices (but we'll call vertices nodes).  Nodes will represent

* data
* operations: addition, multiplication, etc
* activation functions: Softmax, ReLU
* cost functions

Each node will have a value, and the edges which connect nodes, have direction, and will tell us how we pass the values of nodes through the graph.  Consider the following graph

![add](/assets/add.png)

There are three nodes, `X`, `Y`, `Z`.  The nodes `X` and `Y` have edges connecting it to the third node `Z`.  The `+` symbol is meant to show that we are adding the value of `X` and the value of `Y` to get the value of `Z`.  Here `Z` is an outbound node for both `X` and `Y`, while `X` and `Y` are both inbound nodes for `Z`.  To model this, our `Node` class will have a the following fields
* `value` - the value of the node
* `inboundNodes` - a sequence of all nodes feeding into the current node
* `outboundNodes` - a sequence of all nodes the current node feeds into
and the following methods

And so far, one method
* `forward` - tells how to forward values through the graph.  

In scala, we'll model the base class Node as follows  

{% highlight scala %}
package miniflow.nn
class Node(val inboundNodes: Node*) {
  var value: INDArray = null
  val outboundNodes = ArrayBuffer[Node]
  inboundNodes.foreach { n => n.outboundNodes += this }
  def forward(value: INDArray = null) = {
    this.value = value
  }
}
{% endhighlight %}

In terms of the graph we presented above, that would look like

{% highlight scala %}
val x = new Node()
val y = new Node()
val z = new Node(x,y)
{% endhighlight %}

The only issue here is that we have not adequately capture how `X` and `Y` relate to `Z`.  We can improve this with the following classes.

{% highlight scala %}
class Input() exnteds Node()
class Add(x: Node, y: Node) extends Node(x,y) {
  override def forward(value: INDArray = null) = {
    if(value != null) this.value = value
    else this.value = x.value add y.value  
  }
}
{% endhighlight %}

Now, let's revisit creating our graph in scala

{% highlight scala %}
scala> val x = new Input()
x: Input = Input@7e18a0db

scala> val y = new Input()
y: Input = Input@415b4648

scala> val z = new Add(x,y)
z: Add = Add@4dea2521

scala> x.forward(Nd4j.ones(3,3))

scala> y.forward(Nd4j.ones(3,3))

scala> z.forward()

scala> println(z.value)
[[2.00, 2.00, 2.00],
 [2.00, 2.00, 2.00],
 [2.00, 2.00, 2.00]]

{% endhighlight %}

Great! But, I would prefer to write `val z = x + y`, instead of `val z = new Add(x,y)`.  This would give the construction a much nicer feel.  This can be done by adding a `+` method to the `Node` class.

{% highlight scala %}
class Node(val inboundNodes: Node*) {
  ...
  def +(that: Node) = new Add(this, that)
}
{% endhighlight %}

Now we can write `val z = x + y`, which is much nice that `val z = new Add(x,y)`.  You might ask why we didn't just add an add method directly to the Node class instead of create a special Add node.  The reason for this is because we would want to have many methods for a node, each of which would require us to calculate gradients, and it is much easier to modularize the operations as class instead of cramming them all into Node as methods.  

You could go through and create new Node subtypes to represent other operations such as matrix multiplication and hadamard multiplication.  

## Linear Regression Graph

Next, let's consider what a linear regression graph would look like.  We would have two `Input` nodes, one for features and one for labels.  We would need something to represent the weights and bias of the linear regression, for this we'll create a new Node subtype - `Variable`. We'll also create a new node which is meant to represent the linear transformation `Xw + b`, where `X` is an `Input` and `W` and `b` are of type `Variable`

{% highlight scala %}
class Variable() extends Node()

class Linear(x: Input, w: Variable, b: Variable) extends Node(x,w,b) {
  override def forward(value: INDArray = null) = {
    (x.value mmul w.value) addRowVector b.value
  }
}
{% endhighlight %}

Now, consider making the linear regression graph in scala

{% highlight scala %}
scala> val x = new Input()
x: Input = Input@16044d74

scala> val w = new Variable()
w: Variable = Variable@2ee4c080

scala> val b = new Variable()
b: Variable = Variable@5e9b6e6e

scala> val yhat = new Linear(x,w,b)
yhat: Linear = Linear@1951961b

scala> x.forward( Nd4j.randn(10,3))

scala> w.forward( Nd4j.randn(3,1))

scala> b.forward( Nd4j.randn(1,1))

scala> yhat.forward()

scala> println(yhat.value)
[0.07, 0.27, -5.67, -0.66, 0.52, -2.18, 1.41, -4.42, -1.44, -7.95]
{% endhighlight %}

So far so good, but we have omitted a very important part of our model so far - how to calculate derivatives?  This is of utmost importance when we want to start training our neural networks and it will require us to create back propagation methods for all the nodes we want to include in out neural net, e.g., activation functions, cost functions, et cetera.  

Next post will focus on back prop!
