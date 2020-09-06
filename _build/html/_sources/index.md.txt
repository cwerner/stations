```{toctree}
:hidden: true
:maxdepth: 2
:caption: My Title

License <license>
Authors <authors>
Changelog <changelog>
Module Reference <api/modules>
```

```{include} ../README.md
:relative-images:
```

These pages are generated using {doc}`Sphinx documentation <sphinx:usage/quickstart>`.
It is formatted in `reStructuredText` and `markdown`. Add additional pages
by creating md-files in ``docs`` and adding them to the `toctree` above.

It is also possible to refer to the documentation of other Python packages
with the `Python domain syntax`. By default you can reference the
documentation of `Sphinx`, `Python`, `NumPy`, `SciPy`, `matplotlib`,
`Pandas`, `Scikit-Learn`. You can add more by extending the
``intersphinx_mapping`` in your Sphinx's ``conf.py``.

The pretty useful extension `autodoc` is activated by default and lets
you include documentation from docstrings. Docstrings can be written in
`Google style`.

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
