import sys
from sphinx.ext.intersphinx import inspect_main

from conf import intersphinx_mapping


library = sys.argv[1]
url = intersphinx_mapping[library][0] + "/objects.inv"
inspect_main([url])
