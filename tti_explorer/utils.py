from collections import namedtuple
from itertools import product, starmap
import json
import os

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import gamma


def bool_bernoulli(p, rng):
    return bool(rng.binomial(1, p))


def categorical(pvals, rng, n=1):
    outputs = np.argmax(rng.multinomial(1, pvals, size=n), axis=-1)
    return outputs.item() if n == 1 else outputs
# BE, weird to have different output types but also weird to return
# a 1 item array


def he_infection_profile(period, gamma_params):
    inf_days = np.arange(period)
    mass = gamma.cdf(inf_days + 1, **gamma_params) - gamma.cdf(inf_days, **gamma_params)
    return mass / np.sum(mass)


def home_daily_infectivity(base_mass):
    fail_prod = np.cumprod(1 - base_mass)
    fail_prod = np.roll(fail_prod, 1)
    np.put(fail_prod, 0, 1.)
    skewed_mass = fail_prod * base_mass
    return skewed_mass / np.sum(skewed_mass)


def named_product(**items):
    Product = namedtuple('Product', items.keys())
    return starmap(Product, product(*items.values()))


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
            fig.savefig(fpath,**savefig_kwds)


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
