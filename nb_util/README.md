# nbcopy
Tool for making and filling Jupyter notebook templates.

## Usage
	import jupyter_template as jt

Given a Jupyter notebook, we can create a template by replacing specific strings with keywords.

	fname = 'examples/2.mutation-classifier.ipynb'
	new_keywords = {'7157': 'id', 'TP53': 'mutation'}
	template = jt.create_template(fname, new_keywords)

Once we have a template, filling in the keywords is straightforward

	keywords = {'id': 'foo', 'mutation': 'bar'}
	nb = template.fill_template(keywords)
	with open('examples/output.ipynb', 'w') as f:
		f.writelines(nb)
	f.close()
