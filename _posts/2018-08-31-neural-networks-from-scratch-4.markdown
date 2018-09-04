---
layout: post
title:  "Neural Networks from Scrach Part 4"
date:   2018-09-01 22:05:00 -0400
categories: jekyll update
math: true
---

## Vertices

In scala, we typically favor immutable data types, but in some intances, especially this one, it is unavoidable.  Our approach to neural networks in scala is based on creating nodes.  Each node will have set of inbound nodes and outbound nodes.  For our purposes, we will treat the inbound nodes as an immutable sequence, which is created during the construction of the node, and the outbound nodes will be a mutable sequence - only because the nodes our constructed node will feed have not be constructed yet.  

The way we'll handle this, when a node is constructed, we'll add the constructed node to the set of outbound vertex for each inbound vertex.

{% highlight scala %}
class Node(val inboundNodes: Node*) {

  val outboundNodes = ArrayBuffer[Node]

  inboundNodes.foreach { n => n.outboundNodes += this }

}
{% endhighlight %}

And thats it, well... almost.  We'll have to add a `forward()` method as well as a `backward` method to each of the `Node` classes we have created, and we'll also add some other methods and fields for a matter of convenience and neccesity, but overall, the general `Node` class is fairly straightforward.  Once we start thinking about `forward()` and `backward()` we'll also need to consider a `value` field, which will represent the value of the operations up to and including the current node.  The `value` method is what will be forward propagated.  For our purposes, we know the `value` will be a matrix, moreover, is will be an ND4J `INDArray` - so let's go ahead and add that as a field.  As an aside, I like the naming `INDArray` - the `IN` reminds me of how I used to write $$\mathbb{N}$$ in grad school.  

The `forward()` and `backward()` methods will be different for every single type of Node we create, examples of which would be activation nodes, cost function nodes, operations, etc.  

Round two at `Node` class

{% highlight scala %}
import org.nd4j.linalg.api.ndarray.INDArray

class Node(val inboundNodes: Node*) {

  var value: INDArray = null

  val outboundNodes = ArrayBuffer[Node]

  inboundNodes.foreach { n => n.outboundNodes += this }

  def forward(value: INDArray): Unit = {
    if(value != null) this.value = value
  }

  def backward(value: INDArray): Unit = ???

}
{% endhighlight %}

Another convenience would be able to perform some basic operations on our `Node` class, for example `+, *, -, /`.  

{% highlight scala %}
class Add(x: Node, y: Node) extends Node(x,y) {

  def forward(value: INDArray = null): Unit = {
    this.value = x.value + y.value
  }

  def backward(value: INDArray = null): Unit = ???

}
{% endhighlight %}

Now, we could have added this method directly to the `Node` class, but as mentioned earlier, each node we create will need a `forward()` and `backward()` method and to do this all in the `Node` class would make things very difficult.  This choice also makes sense based on how we view the computational graph for the neural network.  

![add](/assets/add.png)

Consider how our api looks now.  

{% highlight scala %}
val x = new Node()
val y = new Node()
val z = new Add(x,y)
{% endhighlight %}

And this is good, but it would be much nicer to be able to write `val z = x + y`.  Gives the construct a much nicer feel.  This can be done by adding the following method to the `Node` class above.  

{% highlight scala %}
class Node(val inboundNodes: Node*) {
  ...
  def +(that: Node) = new Add(this, that)
}
{% endhighlight %}








<!-- Still need to work out an intro to computational graphs (later post)
calculus on computational graph (later post)
general design for the network
 * feedforward
 * backpropagation
mnist
gan on mnist -->
