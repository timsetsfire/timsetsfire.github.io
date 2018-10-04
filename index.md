---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: page
---

<strong>Index</strong>

<nav>
  <h class="nav {{include.navclass}}">
    <li><a href="{{site.baseurl}}/neural-nets.html">Neural Networks From Scratch</a></li>
    <ul>
      Build a neural network from scratch using Scala!  We'll go through picking a linear algebra library, designing the framework and ultimiately build a generative adversarial network to generate data resembling images from MNIST with the framework.  
    </ul>

    <li><a href="{{site.baseurl}}/data-processing.html">Data Processing</a></li>
    <ul style="list-style-type:square">
      Some solutions to problems I have encountered.
      <!-- {% for post in site.categories.data-processing limit: 2%}
        <li>
          <a href="{{post.url}}">{{post.title}}</a>
          <p>{{post.meta}}</p>
        </li>
      {% endfor %} -->
    </ul>
  </h>
</nav>
