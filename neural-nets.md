---
layout: page
---
<strong>Neural Networks</strong>
<nav>
  <ul class="nav {{include.navclass}}">
    <ul>
      {% for post in site.categories.neural-nets reversed %}
      <li>
        <a href="{{post.url}}">{{post.title}}</a>
          <p><font size="-1">{{post.meta}}</font></p>
      </li>
      {% endfor %}
    </ul>
  </ul>
</nav>
<a href="{{site.url}}">Index</a>
