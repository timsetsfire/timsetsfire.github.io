---
layout: post
title:  "Neural Networks from Scrach Part 4"
date:   2018-09-01 22:05:00 -0400
categories: jekyll update
math: true
---

## Feed Forward Neural networks

Here we will begin to get into how we will model our neural network framework.  Per Deep Learning by Goodfellow, the feedforward neural network is called such because

* (feedforward) information flows through the function being evaluated from the input $$x$$, through intermediate computations used to define $$f$$, and finally to the output $$y$$.  We are not considering any feedback connections
* (network) They are typically represented by composing together many different functions.  The model is associated with a directy acyclic graph describing how functions are componsed together.
* (neural) The models are loosly inspired by neuroscience.

We will begin our framework reviewing a simple graph and how we will represent it within our framework.  

![add](/assets/add.png)

There are three nodes, `X`, `Y`, `Z`.  There are two directed edges in this graph, one from `X` to `Z` and one from `Y` to `Z`.  The edges, together with `+` show that we are adding `X` and `Y` to get `Z`.  Here, when we say we are acting on nodes, we are really acting on the values of the node, in this case an n-dimensional array.  Since neither `X` nor `Y` have a parent, it makes sense to refer to them as Input nodes, and based on the relationship between `X`, `Y`, and `Z` it makes sense to call `Z` and Add node.  So, to start our `Node` class will have a the following fields
* `value` - the value of the node
* `inboundNodes` - a sequence of all nodes feeding into the current node
* `outboundNodes` - a sequence of all nodes the current node feeds into

And so far, one method
* `forward` - tells how to forward values through the graph.  In the example of the Add node, the `forward` method would add the values of the inbound nodes.

In scala, we'll model the base class Node, and two classes which extend Node: Input and Add.  

{% highlight scala %}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import scala.collection.mutable.ArrayBuffer

class Node(val inboundNodes: Node*) {
  var value: INDArray = null
  val outboundNodes = ArrayBuffer[Node]()
  inboundNodes.foreach { n => n.outboundNodes += this }
  def forward(value: INDArray = null) = {
    this.value = value
  }
}
class Input() extends Node()
class Add(x: Node, y: Node) extends Node(x,y) {
  override def forward(value: INDArray = null) = {
    if(value != null) this.value = value
    else this.value = x.value add y.value  
  }
}
{% endhighlight %}

In terms of the graph presented above, it's construction in scala would look like

{% highlight scala %}
scala> val x = new Input()
x: Input = Input@b8b1ec8

scala> val y = new Input()
y: Input = Input@5e0cd349

scala> val z = new Add(x,y)
z: Add = Add@363f9b87

scala> x.forward(Nd4j.ones(3,3))

scala> y.forward(Nd4j.ones(3,3).mul(4))

scala> z.forward()

scala> print(z.value)
[[5.00, 5.00, 5.00],
 [5.00, 5.00, 5.00],
 [5.00, 5.00, 5.00]]

scala> z.inboundNodes
res6: Seq[Node] = WrappedArray(Input@b8b1ec8, Input@5e0cd349)

scala> x.outboundNodes
res7: scala.collection.mutable.ArrayBuffer[Node] = ArrayBuffer(Add@363f9b87)

scala> y.outboundNodes
res8: scala.collection.mutable.ArrayBuffer[Node] = ArrayBuffer(Add@363f9b87)
{% endhighlight %}

Great! But, I would prefer to write `val z = x + y`, instead of `val z = new Add(x,y)`.  This would give the construction a much nicer feel.  This can be done by adding a `+` method to the `Node` class.

{% highlight scala %}
class Node(val inboundNodes: Node*) {
  ...
  def +(that: Node) = new Add(this, that)
}
{% endhighlight %}

You might ask why we didn't just add an add method directly to the Node class instead of create a special Add node.  The reason for this is because we would want to have many methods for a node, each of which would require us to calculate gradients, and it is much easier to modularize the operations as class instead of cramming them all into Node as methods.  

We'll be using the node model to represent

* inputs
* operations: addition, multiplication, etc
* activation functions: Softmax, ReLU, etc
* cost functions


## Linear Regression Graph

Next, let's consider what a linear regression graph would look like.  We would have two `Input` nodes, one for features and one for labels.  We would need something to represent the weights and bias of the linear regression, for this we'll create a new Node subtype - `Variable`. We'll also create a new node which is meant to represent the linear transformation `Xw + b`, where `X` is of type `Input` and `W` and `b` are of type `Variable`, and finally, a Cost function, which will take our actual and predicted to assess loss - the obvious choice is MSE.   

{% highlight scala %}
class Variable() extends Node()

class Linear(x: Input, w: Variable, b: Variable) extends Node(x,w,b) {
  override def forward(value: INDArray = null) = {
    this.value = (x.value mmul w.value) addRowVector b.value
  }
}

class Mse(y: Input, yhat: Node) extends Node(y,yhat){
  override def forward(value: INDArray = null): Unit = {
    val obs = y.value.shape.apply(0).toDouble
    val diff = y.value - yhat.value
    this.value = (diff * diff).sum(0).sum(1) / (obs.toDouble)
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

scala> val mse = new Mse(y, yhat)
mse: Mse = Mse@6ea57b84

scala> x.forward( Nd4j.randn(10,3))

scala> w.forward( Nd4j.randn(3,1))

scala> b.forward( Nd4j.randn(1,1))

scala> yhat.forward()

scala> mse.forward()

scala> println(y.value)
[-1.25, -0.94, -0.82, -1.34, -0.61, 0.14, -0.83, -0.80, -1.00, -0.37]

scala> println(yhat.value)
[-3.05, -1.38, -0.73, -1.41, -0.84, 0.21, 1.40, 1.58, -0.59, -2.55]

scala> println(mse.value)
1.91
{% endhighlight %}

So far so good, with one exception - we haven't talked about how to learn this graph, i.e., learn the parameters of the graph.  The next post we'll talk about automatic differentiation and how we'll handle this in our framework.  This will help us figure out how to calculate the gradients for any optimization technique that can be used to learn the graph.  

Cheers!
