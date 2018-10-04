---
layout: page
---

As a data scientist, I spent a significant amount of time manipulating data.  I have gathered some problems I have encountered and their solutions in the hope that it will be of some use to others.  

<strong>Data Processing</strong>
<ul style="list-style-type:square">
  {% for post in site.categories.data-processing limit: 2%}
    <li>
      <a href="{{post.url}}">{{post.title}}</a>
      <p>{{post.meta}}</p>
    </li>
  {% endfor %}
</ul>
<a href="{{site.url}}">Index</a>
