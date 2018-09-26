---
layout: post
title:  "Back Propagation"
date:   2018-09-12 22:05:00 -0400
categories: neural networks from scratch in scala
series: neural_networks
math: true
---

## Function composition
<!--
<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul> -->


Function composition can be easily understood as the application of one function to the result of another function to produce a third function.  

Concretely, Suppose that we have a function $h = (g \circ f)(x)$, read $g$ composed with $f$ of $x$, with $f: X \to Y, g: Y \to Z$, then $h: X \to Z$.  

## Function Composition and Feed Forward Networks

![network2](/assets/graph.png)

Consider the simple network above -
* The Input and Variable node feed into a Multiply node.  This implies that the value of the Multiply node is obtained by multiplying the value of the Input node and the Variable node.  
* The Multiply nodes feeds into an Activation node, which, along with another Variable node feed into a Mutlipy Node, et cetera.  

The way the values feed forward through this graph is nothing more than evaluation of many function compositions.  What is still not clear is how we learn this graph, which, in this context refers to estimating the parameters of the network (i.e., the values of the Variable nodes).  The standard way to learn neural networks is via gradient based optimization methods, which, obviously, requires us to compute gradients for our trainable parameters, which means we need to pay a visit to our old friend the chain rule!

## Chain rule

If we have a function $$(h \circ g \circ f ) (x)$$, we will calc its derivative via the chain rule as

$$\frac{\partial h}{\partial x} = \frac{\partial g}{\partial f}\frac{\partial f}{\partial x}$$

Here we are assuming the $f$ and $g$ are differentiable on their domain, thus $h$ is differentiable on its domain.  

## Automatic Differentiation

Straight from wikipedia

> [Automatic Differentiation] exploits the fact that every computer program, no matter how complicated, executes a sequence of elementary arithmetic operations (addition, subtraction, multiplication, division, etc.) and elementary functions (exp, log, sin, cos, etc.). By applying the chain rule repeatedly to these operations, derivatives of arbitrary order can be computed automatically, accurately to working precision, and using at most a small constant factor more arithmetic operations than the original program.

We can approach the evaluation of the derivative via automatic differentiation in one of two ways

* Forward accumulation - traverse chain rule inside out
* Reverse accumulation - traverse chain rule from outside in

<!-- In forward accumulation, you pick the variable with which you will differentiate with respect to, which means you traverses the chain rule from inside out.  In reverse accumulation, you pick the variable which will be differentiated thus

Back propagation is a special case of reverse accumulation.  

In function that we just described, going forward or in reverse really would not make a difference, but in a larger, more complex graph, you will realize significant speed up by use the reverse accumulation approach rather than the forward for the following reason
Forward accumulation will give the derivation of the output with respect to a single node, while reverse accumulation gives the derivative of the output with respect to all the nodes.  Check out [this great post on the topic](http://colah.github.io/posts/2015-08-Backprop/).  This reverse accumulation is exactly back propagation!!! -->


## Backpropagation

Back propagation is used in neural networks to calculate gradients, which will help us train the neural network using gradient based optimization techniques.  

In order to understrand how to use backprop, we first need to visit our old friend the chain rule from calculus.  The chain rule is used to calculate derivatives of functions composed with functions.  Concretely, if we have $$h(x) = (g \circ f)(x) = g( f(x) )$$, we can calulate the derivative as

$$h'(x) = g'(f(x))f'(x)$$

Function composition is exactly how we built our neural network.  We have an input $$x$$ and several hidden layers, which are defined by linear transformations and elementwise operations, which finally terminate at an output value $$\hat{y}$$ which will be compared against our target column via some cost function.  

Consider this simple example

![network](/assets/network.png)
![network2](/assets/networkgraph.png)

$$\begin{align*}
Z_1 &= X\beta \\
H_1 &= \phi_1(Z_1) \\
Z_2 &= H_1\beta_2 \\
\hat{Y} &= \phi(Z_2) \\
MSE &= \frac{1}{2m} \sum (Y - \hat{Y})^2
\end{align*}$$

Here $$X$$ is our feature set of dimension $m \times n$, $$Y$$ is the target with dimension $$m \times p$$, $$\phi_1, \phi_2: \mathbb{R} \to \mathbb{R}$$ are called Activation Functions and are assumed to be differentiable almost everywhere.  To say that it is differentiable almost everywhere is to say that the set of element at which the functions aren't differentiable is countable.  

Now if we want to calculate the gradient of $$b_1$$, we will go backward from the terminal node, backward to $$b_1$$.  

$$\frac{\partial MSE}{ \partial \beta_1} = \frac{\partial MSE}{\partial H_2} \frac{\partial H_2}{\partial Z_2}\frac{\partial Z_2}{\partial H_1}\frac{\partial H_1}{\partial Z_1}\frac{\partial Z_1}{\partial \beta_1}.$$

For the most part, the derivatives we must calculate are very easy.  Since the Activation functions are elementwise, so are the derivatives.  The cost function is fairly straight forward in my opinion.  The only one that may give us any trouble is the derivative of the linear transformation.  

## Activation functions

These are all elementwise and differentiable almost everywhere, so if $Y = \phi(X)$ then $\partial Y / \partial X = \phi'(X)$.  


## Mean Square Error

This is fairly straight forward.  Suppose that $Y, \hat{Y}$ are $m \times n$ matrices, then $$MSE = \frac{1}{2m}\sum\sum (Y - \hat{Y})^2$$, then we'll calc the partials as follows

$$\frac{\partial MSE}{\partial Y_{i,j}}  = \frac{1}{m}(Y_{i,j} - \hat{Y}_{i,j})$$

Thus,

$$\frac{\partial MSE}{\partial Y} = \frac{1}{m}(Y - \hat{Y})$$

it follows

$$\frac{\partial MSE}{\partial \hat{Y}} = \frac{-1}{m}(Y - \hat{Y})$$

## Linear Node

I always took this derivative for granted.  It was until I sat down to write this post did I do these calculations by hand and realized they suck!  Consider $$Y = X\beta + {\bf 1}_{m}a$$ where $X$ is $m \times n$ and $\beta$ is $n \times p$ dimension, $a$ is $1 \times p$, ${\bf 1}_m$ is a column of 1's with dimension $m \times 1$, and finally $Y$ is $m \times p$.  

Let $y = vec(Y)$ be the stacking of columns of $Y$ into a column vector, which has dimension $m\cdot p \times 1$, then the following holds

$$vec(Y) = (\beta^T \otimes I_m) vec(X)$$

and

$$vec(Y) = (I_p \otimes X) vec(\beta)$$

Now, if $y = vec(Y), x = vec(X), b = vec(\beta)$, then

$$\frac{\partial y}{\partial x} = \beta^T \otimes I_m$$

and

$$\frac{\partial y}{\partial b} = I_p \otimes X$$

For completeness, we should consider the linear transformation

$$Y = X\beta + 1_{m}a$$

Here $a$ is a $1 \times p$ vector of biases.  Using the same rules as above, we get

$$\frac{\partial vec(Y)}{ \partial a^T}  = I_{p} \otimes 1_{m}$$


As is, this isn't bad to use in the back prop, but we'll be afforded some conveniences based on the the maths that simplify the cacluations during back prop which will not require use to compute those nasty Kronecker products.  To see this, consider $$H = \phi(X\beta)$$, with $\phi$ an elementwise operation, $h = vec(H), b = vec(\beta)$ and $x = vec(X)$, we can use the properties listed above, and the fact that that $vec(\phi(X\beta)) = \phi(vec(X\beta)),$ to simplify these calcs

$$\begin{align*}
\frac{\partial h}{ \partial x} &= (\beta^T \otimes I)^T \phi'(vec(X\beta)) \\
&= (\beta \otimes I) \phi'(vec(X\beta))\\
\end{align*}
$$

and

$$\begin{align*}
\frac{\partial h}{\partial b} &= (I \otimes X)^T \phi'(vec(X\beta)) \\
&= (I \otimes X^T)\phi'(vec(X\beta)) \\
\end{align*}
$$

Un-vectorizing these result gives

$$\partial H / \partial X = \phi'(X\beta)\beta^T$$

 and

 $$\partial H / \partial \beta = X^T \phi'(X\beta)!$$

So if $\phi$ were equal to $(y - X\beta)^2$, it is clear how differentiating this with respect to $X$ and $\beta$ result in the $-X^T(y - X\beta)$ and $-(y - X\beta)\beta^T$ respectively.  

{% include series.html %}
