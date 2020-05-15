from collections import namedtuple
from itertools import product, starmap
import json
import os
import pathlib

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


ROOT_DIR = pathlib.Path(__file__).parent.parent


def bool_bernoulli(p, rng):
    return bool(rng.binomial(1, p))


def categorical(pvals, rng, n=1):
    """categorical

    Args:
        pvals (iterable[float]): probabilities
        rng (np.random.RandomState): random state from which to draw
        n (int): Number of iid samples

    Returns:
        result (int or np.array[int]): depending on whether `n`==1
    """
    outputs = np.argmax(rng.multinomial(1, pvals, size=n), axis=-1)
    return outputs.item() if n == 1 else outputs
# BE, weird to have different output types but also weird to return
# a 1 item array


def load_cases(fpath):
    """load_cases
    Loads case and contact from .json file into Cases and Contacts.

    Args:
        fpath (str): path to file.

    Returns (tuple[list[tuple[Case, Contact], dict]):
        pairs: list of Case, Contact pairs
        meta: dictionary of meta-data for case/contact generation
    """
    from tti_explorer.case import Case
    from tti_explorer.contacts import Contacts, NCOLS

    with open(fpath, "r") as f:
        raw = json.load(f)

    cases = raw.pop("cases")
    meta = raw
    pairs = list()
    for dct in cases:
        case = Case(**dct['case'])
        contacts_dct = dct['contacts']
        n_daily = contacts_dct.pop('n_daily')
        contacts_dct = {
                k: np.array(v, dtype=int).reshape(-1, NCOLS)
                for k, v in contacts_dct.items()
                }
        contacts = Contacts(n_daily=n_daily, **contacts_dct)
        pairs.append((case, contacts))
    return pairs, meta


def named_product(**items):
    Product = namedtuple('Product', items.keys())
    return starmap(Product, product(*items.values()))


def swaplevel(dct_of_dct):
    keys = next(iter(dct_of_dct.values())).keys()
    return {in_k: {out_k: v[in_k] for out_k, v in dct_of_dct.items()} for in_k in keys}


def read_json(fpath):
    with open(fpath, "r") as f:
        return json.loads(f.read())


def write_json(stuff, fpath):
    with open(fpath, "w") as f:
        return json.dump(stuff, f)


def sort_by(lst, by, return_idx=False):
    idx, res = zip(*sorted(zip(by, lst)))
    return (res, idx) if return_idx else res


class Registry:
    "Case insensitive registry"
    def __init__(self):
        self._register = dict()

    def __getitem__(self, key):
        return self._register[key.lower()]

    def __call__(self, name):
        def add(thing):
            self._register[name.lower()] = thing
            return thing
        return add


class PdfDeck:
    def __init__(self, figs=None, names=None):
        self.figs = figs or []
        self.fignames = names or []

    @classmethod
    def save_as_pdf(cls, figs, fpath):
        return cls(figs).make(fpath)
    
    def default_figname(self):
        return f"{repr(self).replace(' ', '_')}_figure_{len(self.figs)}"
    
    def add_figure(self, fig, *, position=None, name=None):
        if position is None:
            self.figs.append(fig)
            self.fignames.append(name or self.default_figname())
        else:
            self.figs.insert(position, fig)

    def make(self, fpath):
        with PdfPages(fpath) as pdf:
            for fig in self.figs:
                pdf.savefig(fig)
    
    def make_individual(self, folder=None, **savefig_kwds):
        folder = folder or os.cwd()
        for fig, name in zip(self.figs, self.fignames):
            fpath = os.path.join(folder, name+"."+savefig_kwds.get("format", "pdf"))
            fig.savefig(fpath, **savefig_kwds)


class LatexTableDeck:
    table_template = r"""
    \begin{table}[H]
         %(table)s
        \caption{%(caption)s}
    \end{table}
    """

    header = r"""

    \documentclass{article}

    \usepackage{booktabs}
    \usepackage{tabularx}
    \usepackage{float}

    \restylefloat{table}

    \begin{document}

    """
    clearpage_str = "\clearpage"
    footer = "\n\end{document}"
    new_section = r"\section{%s}"
    
    def __init__(self, table_template=None, header=None, footer=None, new_section=None, clearpage_str=None):
        self.table_template = table_template or self.table_template
        self.header = header or self.header
        self.footer = footer or self.footer
        self.new_section = new_section or self.new_section
        self.clearpage_str = clearpage_str or self.clearpage_str

        self.strings = list()
    
    def add_section(self, section_name):
        self.strings.append(self.new_section % section_name)
        
    def add_table(self, tex_table, caption):
        self.strings.append(self.table_template % dict(table=tex_table, caption=caption))
        
    def clearpage(self):
        self.strings.append(self.clearpage_str)
    
    def make(self, fpath, joiner="\n\n"):
        output = joiner.join([self.header, *self.strings, self.footer])
        with open(fpath, "w") as f:
            f.write(output)
