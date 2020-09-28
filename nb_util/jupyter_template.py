# -*- coding: utf-8 -*-
"""Lightweight templates for Jupyter notebooks."""

import nbformat
import re
from collections import namedtuple
# consider looking into nbconvert


Cell = namedtuple('Cell', ['type', 'source'])
Cell.__doc__ = """A Jupyter notebook cell.

type - type of the cell
source - contents of the cell
"""

__new_cell = {
    'code': nbformat.v4.new_code_cell,
    'markdown': nbformat.v4.new_markdown_cell,
    'raw': nbformat.v4.new_raw_cell,
}


def each_cell(func):
    """Extend a function defined on a source cell to work on a notebook of
    cells.

    Args:
    func - function which maps str to str

    """
    return lambda cells: [Cell(c.type, func(c.source)) for c in cells]


class JupyterTemplate(object):
    """
    Represent a Jupyter notebook template.

    Instance variables:
    cells - iterator of Cell
    keywords - list of strings

    Public methods:
    fill_template - raises a KeyError if the dictionary doesn't contain the
    proper keys.

    """
    def __init__(self, cells, keywords):
        """Create a JupyterTemplate.

        Keyword argumnets:
        cells - iterator of Cell
        keywords - iterator of str

        """
        self.cells = cells
        self.keywords = sorted(keywords)

    @classmethod
    def _get_cells_from_nb(cls, fname):
        """
        Extract cells from a file.

        Args:
        fname - file name

        """
        nb = nbformat.read(fname, nbformat.NO_CONVERT)
        return [Cell(c['cell_type'], c['source']) for c in nb['cells']]

    def fill_template(self, keywords):
        """Replace keywords to create a template.

        Args:
        keywords - dictionary of strings

        """
        if sorted(keywords.keys()) != self.keywords:
            raise KeyError('Expected keys: {}'.format(self.keywords))
        substitute = _substitute(keywords)
        new_cells = substitute(self.cells)
        new_cells = _decode(new_cells)
        return _create_nb(new_cells)


@each_cell
def _encode(s):
    """Encode source cells.

    Args:
    s - jupyter notebook cell source

    """
    return s.replace('{', '{{').replace('}', '}}')


@each_cell
def _decode(s):
    """Decode source cells.

    Args:
    s - jupyter notebook cell source

    """
    return s.replace('{{', '{').replace('}}', '}')


def _substitute(values):
    """Substitute values into each source cell.

    Args:
    values - dict of the form {str: str}

    """
    return each_cell(lambda s: s.format(**values))


def _replace(keywords):
    """Replace keywords with specified values.

    Args:
        keywords (dict of str: str):

    """
    def f(s):
        for k, v in keywords.items():
            s = re.sub(r'\b' + k + r'\b', '{' + v + '}', s, count=0)
        return s
    return each_cell(f)


def create_template(fname, new_keywords):
    """Replace keywords to create a template.

    Args:
    fname - file namedtuple
    keywords - dict whose keys

    """
    cells = JupyterTemplate._get_cells_from_nb(fname)
    new_cells = _encode(cells)
    new_cells = _replace(new_keywords)(new_cells)
    return JupyterTemplate(new_cells, new_keywords.values())


def _create_nb(cells):
    """Return a Jupyter notebook as a string.

    Args:
        cells - iterator of Cells

    """
    nbjson = nbformat.v4.new_notebook()
    nbjson.cells = [__new_cell[c.type](c.source) for c in cells]
    return nbformat.writes(nbjson, version=nbformat.current_nbformat)
