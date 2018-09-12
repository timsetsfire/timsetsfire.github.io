---
layout: post
title:  "Neural Networks from Scrach Part 5"
date:   2018-09-15 22:05:00 -0400
categories: jekyll update
math: true
---

## Function composition

Function composition is the backbone of our neural network architecture.  It can be easily understood as the application of one function to the result of another function to produce a third function.  

Concretely, Suppose that we have a function $h = (g \circ f)(x)$, read $g$ composed with $f$ of $x$.  It is understood, that $f: X \to Y$, g: Y \to Z$, then $h: X \to Z$.  In the previous post, we used function composition to formalize the architecture for our neural network.  Moreover, we discussed how to feed data through our network (forward) and in doing so, we were doing the evaluation of a functions composed with functions.  As we left off with our linear regression graph, we still had not touched on an important part - how to learn the graph.  In this context learn is referrring to estimating the parameters of the network.  The standard way to learn neural networks is via gradient based optimization methods, which, obviously, requires us to compute gradients for our trainable parameters, which means we need to pay a visit to our old friend the chain rule!

## Chain rule

If we have a function $$(h \circ g \circ f ) (x)$$, we will calc its derivative via the chain rule as

$$\frac{\partial h}{\partial x} = \frac{\partial g}{\partial f}\frac{\partial f}{\partial x}$$

Here we are assuming the $f$ and $g$ are differentiable on their domain, thus $h$ is differentiable on its domain.  

## Automatic Differentiation

Straight from wikipedia

> [Automatic Differentiation] exploits the fact that every computer program, no matter how complicated, executes a sequence of elementary arithmetic operations (addition, subtraction, multiplication, division, etc.) and elementary functions (exp, log, sin, cos, etc.). By applying the chain rule repeatedly to these operations, derivatives of arbitrary order can be computed automatically, accurately to working precision, and using at most a small constant factor more arithmetic operations than the original program.

Consider the multivariate example $h(x,b) = \phi(f(x,b))$.  

Assuming that $\phi$ and $f$ are differentiable on their domain, we can calculate partials as follows

$$\frac{\partial h}{\partial x} = \frac{\partial \phi}{\partial f}\frac{\partial f}{\partial b}$$

and

$$\frac{\partial h}{\partial b} = \frac{\partial \phi}{\partial f}\frac{\partial f}{\partial x}$$

One thing worth pointing our at this point, both partial derivatives above involve a similar calculation, in particular $\frac{\partial \phi}{\partial f}$.  Now suppose that we have fixed values of $x$ and $b$, say $x'$ and $b'$.  We would have forwarded these values through the composition, stored all intermediate calculations, and then put everything together.  

This is very easy application of the chain rule, but when we begin to compute gradients within our neural network, the calculations will not be so easy and we will utilize the method of automatic differentiation.  

We can approach the evaluation of the derivative vai automatic differentiation in one of two ways

* Forward accumulation
* Reverse accumulation

With forward accumulation we would first calc $$\partial f / \partial x$$ evaluated as $x',b'$ followed by $$\partial  \phi / \partial f$$ evaluated at $x', b'$, while reverse accumulation would be, well, the reverse, starting with $\partial \phi / \partial f$ evaluated at $x', b'$ then $\partial f / \partial x$ evaluated at $x', b'$.  

One thing to keep in mind - we already forwarded $x', b'$ through the composition, so we have stored $f(x,b)$ and $\phi(f(x,b))$

### Forward accumulation example

$$\frac{\partial h}{\partial x}\bigg|_{x = \pi} = f'(\pi) g'(f(2\pi))$$

The only new calc here is $f'(\pi)$ and $g'(2\pi)$

### Backward accumulation example

$$\frac{\partial h}{\partial x}\bigg|_{x = \pi} = g'(f(2\pi))f'(\pi) $$

Same story here, the only new calc here is $f'(\pi)$ and $g'(2\pi)$

For this example, it is not clear what the difference is - both methods result in the same number of calculations, but, in larger, more complex graphs, we will realize significant speed up by using the reverse method (aka back propagation)!

Thing about it like this, suppose you have $h(y) = \phi(y), y = xb$. Computing this derivative via forward accumulation, you would calc $\phi'(y)$ twice, once for $\partial h / \partial x$ and once for $\partial h / \partial b$, but the backward method would only have you calcing $\phi'(y)$ onces


In function that we just described, going forward or in reverse really would not make a difference, but in a larger, more complex graph, you will realize significant speed up by use the reverse accumulation approach rather than the forward for the following reason
Forward accumulation will give the derivation of the output with respect to a single node, while reverse accumulation gives the derivative of the output with respect to all the nodes.  Check out [this great post on the topic](http://colah.github.io/posts/2015-08-Backprop/).  This reverse accumulation is exactly back propagation!!!


## Backpropagation

Back propagation is used in neural networks to calculate gradients, which will help us train the neural network using gradient based optimization techniques.  

In order to understrand how to use backprop, we first need to visit our old friend the chain rule from calculus.  The chain rule is used to calculate derivatives of functions composed with functions.  Concretely, if we have $$h(x) = (g \circ f)(x) = g( f(x) )$$, we can calulate the derivative as

$$h'(x) = g'(f(x))f'(x)$$

Function composition is exactly how we built our neural network.  We have an input $$x$$ and several hidden layers, which are defined by linear transformations and elementwise operations, which finally terminate at an output value $$\hat{y}$$ which will be compared against our target column via some cost function.  

Consider this simple example

![network](/assets/network.png)

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

This is fairly straight forward.  If $$MSE = \frac{1}{2m}\sum (Y - \hat{Y})^2$$, then we'll calc the partials as follows

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

Un-vectorizing these result gives $\partial H / \partial X = \phi'(X\beta)\beta^T$ and $\partial H / \partial \beta = X^T \phi'(X\beta)$!

So if $\phi$ were equal to $(y - X\beta)^2$, it is clear how differentiating this with respect to $X$ and $\beta$ result in the $-X^T(y - X\beta)$ and $-(y - X\beta)\beta^T$ respectively.  
