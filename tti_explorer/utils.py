from collections import namedtuple
from itertools import product, starmap
import json
import os
import pathlib

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


ROOT_DIR = pathlib.Path(__file__).parent.parent

CASE_KEY = "case"
CONTACTS_KEY = "contacts"


def bool_bernoulli(p, rng):
    return rng.uniform() < p
    # return bool(rng.binomial(1, p))


def categorical(pvals, rng, n=1):
    """ Sample from categories according to their probabilities.

    Args:
        pvals (iterable[float]): probabilities of each category, should sum to 1
        rng (np.random.RandomState): random state from which to draw
        n (int): Number of iid samples, defaults to 1

    Returns:
        Indexes of drawn categories, np.array[int] of shape (n,)
    """
    outputs = np.argmax(rng.multinomial(1, pvals, size=n), axis=-1)
    return outputs


def load_cases(fpath):
    """load_cases
    Loads case and contact from .json file into Cases and Contacts.

    Args:
        fpath (str): path to file.

    Return:
        Tuple of:
            pairs: list of tuples (Case, Contacts)
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
        case = Case(**dct[CASE_KEY])
        contacts_dct = dct[CONTACTS_KEY]
        n_daily = contacts_dct.pop("n_daily")
        contacts_dct = {
            k: np.array(v, dtype=int).reshape(-1, NCOLS)
            for k, v in contacts_dct.items()
        }
        contacts = Contacts(n_daily=n_daily, **contacts_dct)
        pairs.append((case, contacts))
    return pairs, meta


def named_product(**items):
    Product = namedtuple("Product", items.keys())
    return starmap(Product, product(*items.values()))


def swaplevel(dct_of_dct):
    keys = next(iter(dct_of_dct.values())).keys()
    return {in_k: {out_k: v[in_k] for out_k, v in dct_of_dct.items()} for in_k in keys}


def map_lowest(func, dct):
    return {
        k: map_lowest(func, v) if isinstance(v, dict) else func(v)
        for k, v in dct.items()
    }


def read_json(fpath):
    with open(fpath, "r") as f:
        return json.loads(f.read())


def write_json(stuff, fpath):
    with open(fpath, "w") as f:
        return json.dump(stuff, f)


def sort_by(lst, by, return_idx=False):
    idx, res = zip(*sorted(zip(by, lst)))
    return (res, idx) if return_idx else res


def get_sub_dictionary(adict, keys):
    return {k: adict[k] for k in keys if k in adict}


def find_case_files(folder, ending=".json"):
    return list(filter(lambda x: x.endswith(ending), os.listdir(folder)))


def tidy_fname(fname, ending=".json"):
    return fname.rstrip(ending)


class Registry:
    "Case insensitive registry"

    def __init__(self):
        self._register = dict()

    def __getitem__(self, key):
        if key.lower() not in self._register:
            raise ValueError(f"{key} isn't registered")
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
            fpath = os.path.join(folder, name + "." + savefig_kwds.get("format", "pdf"))
            fig.savefig(fpath, **savefig_kwds)


# TODO: Make it so you can change the templates in here
class LatexTableDeck:
    table_template = r"""
    \begin{table}[H]
        \centering
         %(table)s
        \caption{%(caption)s}
    \end{table}
    """

    header = r"""

    \documentclass{article}

    %(packages)s

    \restylefloat{table}

    \begin{document}

    """

    clearpage_str = r"\clearpage"
    footer = "\n" + r"\end{document}"
    new_section = r"\section{%s}"

    def __init__(
        self,
        table_template=None,
        header=None,
        footer=None,
        new_section=None,
        clearpage_str=None,
    ):
        self.table_template = table_template or self.table_template
        self.header = header or self.header
        self.footer = footer or self.footer
        self.new_section = new_section or self.new_section
        self.clearpage_str = clearpage_str or self.clearpage_str

        self.strings = list()
        self.packages = [
            r"\usepackage{booktabs}",
            r"\usepackage{tabularx}",
            r"\usepackage{float}",
        ]

    def add_section(self, section_name):
        self.strings.append(self.new_section % section_name)

    def add_string(self, string):
        self.strings.append(string)

    def add_table(self, tex_table, caption):
        self.strings.append(
            self.table_template % dict(table=tex_table, caption=caption)
        )

    def add_package(self, package, options=None):
        pstr = r"\usepackage"
        if options is not None:
            pstr += f"[{', '.join(options)}]"
        pstr += f"{{{package}}}"
        self.packages.append(pstr)

    def clearpage(self):
        self.strings.append(self.clearpage_str)

    def _make_header(self):
        return self.header % {"packages": "\n".join(self.packages)}

    def to_str(self, joiner="\n"):
        return joiner.join([self._make_header(), *self.strings, self.footer])

    def __str__(self):
        return self.to_str(joiner="\n")

    def make(self, fpath, joiner="\n"):
        with open(fpath, "w") as f:
            f.write(self.to_str(joiner=joiner))
