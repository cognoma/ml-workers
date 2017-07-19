import pytest
import nb_util.jupyter_template as jt
from unittest.mock import Mock


@pytest.fixture()
def template_cells():
    return [jt.Cell('markdown', '# Addition'),
            jt.Cell('code', 'x, y = {var1}, {var2}'),
            jt.Cell('code', "{{'ans': x + y}}")]


@pytest.fixture()
def filled_template():
    return """{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Addition"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "x, y = 1, 2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "{'ans': x + y}"
   ]
  }
 ],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 2
}"""


@pytest.fixture()
def new_values():
    return {'var1': 1, 'var2': 2}


@pytest.fixture()
def keywords():
    return ['var1', 'var2']


@pytest.fixture()
def new_keywords():
    return {'x': 'var1', 'y': 'var2'}


@pytest.fixture
def small_cells():
    return [jt.Cell('markdown', '# Tiny document {x} y xy')]


@pytest.fixture
def small_notebook():
    return """{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Tiny document {x} y xy"
   ]
  }
 ],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 2
}"""


@pytest.fixture
def small_template():
    return jt.JupyterTemplate(
        cells=[jt.Cell('markdown',
                       '# Tiny document {{{var1}}} {var2} xy')],
        keywords=['var1', 'var2'])


def test_JupyterTemplate_init_(template_cells, keywords):
    template = jt.JupyterTemplate(template_cells, keywords)
    assert template.cells == template_cells
    assert template.keywords == keywords


def test_create_template(small_cells, new_keywords, small_template,
                         monkeypatch):
    fake_read = Mock(return_value=small_cells)
    monkeypatch.setattr(jt.JupyterTemplate, '_get_cells_from_nb', fake_read)
    new_template = jt.create_template('filename', new_keywords)
    assert new_template.cells == small_template.cells
    assert new_template.keywords == small_template.keywords


def test_fill_template(template_cells, keywords, new_values, filled_template):
    template = jt.JupyterTemplate(template_cells, keywords)
    assert template.fill_template(new_values) == filled_template


def test_encode(small_cells):
    new_cells = jt._encode(small_cells)
    assert new_cells == [jt.Cell('markdown', '# Tiny document {{x}} y xy')]


def test_decode(small_cells):
    new_cells = jt._decode(small_cells)
    assert new_cells == small_cells


def test_encode_and_decode(small_cells):
    new_cells = jt._encode(small_cells)
    assert small_cells == jt._decode(new_cells)


def test_substitute(new_values, template_cells):
    new_cells = jt._substitute(new_values)(template_cells)
    assert new_cells == [jt.Cell('markdown', '# Addition'),
                         jt.Cell('code', 'x, y = 1, 2'),
                         jt.Cell('code', "{'ans': x + y}")]


def test_replace(new_keywords, small_cells):
    new_cells = jt._replace(new_keywords)(small_cells)
    assert new_cells == [jt.Cell('markdown',
                         '# Tiny document {{var1}} {var2} xy')]


def test_create_nb(small_cells, small_notebook):
    nb = jt.create_nb(small_cells)
    assert nb == small_notebook
