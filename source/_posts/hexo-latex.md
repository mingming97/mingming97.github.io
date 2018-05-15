---
title: hexo配置LaTeX公式
date: 2018-5-15 21:12:18
tags: [LaTeX, hexo]
categories: hexo
---

&emsp;&emsp;正常的hexo框架在默认情况下渲染数学公式会有很多问题，可以通过将hexo默认的引擎“hexo-renderer-marked”更换为“hexo-renderer-markdown-it-plus”来渲染markdown。
&emsp;&emsp;首先要将之前的“hexo-renderer-marked”卸载，并安装“hexo-renderer-markdown-it-plus”。
```
npm uninstall hexo-renderer-marked --save
npm install hexo-renderer-markdown-it-plus --save
```
&emsp;&emsp;在这之后建议修改根目录下的\_config.yml，添加如下内容。
```
markdown_it_plus:
  highlight: true
  html: true
  xhtmlOut: true
  breaks: true
  langPrefix:
  linkify: true
  typographer:
  quotes: “”‘’
  plugins:
    - plugin:
        name: markdown-it-katex
        enable: true
    - plugin:
        name: markdown-it-mark
        enable: false
```
&emsp;&emsp;在文章中要启用mathjax，可以在markdown的YAML Front Matter处添加mathjax: true来解决，也可以在主题配置文件里去解决。以next为例，在./themes/next/\_config.yml中，将mathjax的enable项设置为true，就可以不用每次都加入mathjax的设置了。

