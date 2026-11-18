"""Curated bibliography for DrugHX reports (BibTeX / RIS export)."""
from __future__ import annotations

# Keep this list stable and human-curated: it is exported verbatim to reference
# managers.  Every entry must be a real, verifiable publication.
REFERENCES = [
    {
        "key": "eberhardt2021vina",
        "type": "article",
        "authors": ["Eberhardt, J.", "Santos-Martins, D.", "Tillack, A. F.", "Forli, S."],
        "title": "AutoDock Vina 1.2.0: New Docking Methods, Expanded Force Field, and Python Bindings",
        "journal": "Journal of Chemical Information and Modeling",
        "year": 2021,
        "volume": "61",
        "number": "8",
        "pages": "3891--3898",
        "doi": "10.1021/acs.jcim.1c00203",
    },
    {
        "key": "trott2010vina",
        "type": "article",
        "authors": ["Trott, O.", "Olson, A. J."],
        "title": "AutoDock Vina: Improving the Speed and Accuracy of Docking with a New Scoring Function, Efficient Optimization, and Multithreading",
        "journal": "Journal of Computational Chemistry",
        "year": 2010,
        "volume": "31",
        "number": "2",
        "pages": "455--461",
        "doi": "10.1002/jcc.21334",
    },
    {
        "key": "ding2023vinagpu",
        "type": "article",
        "authors": ["Ding, J.", "Tang, S.", "Mei, Z.", "Wang, L.", "Huang, Q.", "Hu, H.", "Ling, M.", "Fu, J."],
        "title": "Vina-GPU 2.0: Further Accelerating AutoDock Vina and Its Derivatives with Graphics Processing Units",
        "journal": "Journal of Chemical Information and Modeling",
        "year": 2023,
        "volume": "63",
        "number": "7",
        "pages": "1982--1998",
        "doi": "10.1021/acs.jcim.2c01504",
    },
    {
        "key": "koes2013smina",
        "type": "article",
        "authors": ["Koes, D. R.", "Baumgartner, M. P.", "Camacho, C. J."],
        "title": "Lessons Learned from Comparing Molecular Docking Poses on the PDBbind Dataset",
        "journal": "Journal of Chemical Information and Modeling",
        "year": 2013,
        "volume": "53",
        "number": "8",
        "pages": "1893--1904",
        "doi": "10.1021/ci300604z",
    },
    {
        "key": "alhossary2015qvina2",
        "type": "article",
        "authors": ["Alhossary, A.", "Handoko, S. D.", "Mu, Y.", "Kwoh, C.-K."],
        "title": "Fast, Accurate, and Reliable Molecular Docking with QuickVina 2",
        "journal": "Bioinformatics",
        "year": 2015,
        "volume": "31",
        "number": "13",
        "pages": "2214--2216",
        "doi": "10.1093/bioinformatics/btv082",
    },
    {
        "key": "gaulton2017chembl",
        "type": "article",
        "authors": ["Gaulton, A.", "Hersey, A.", "Nowotka, M.", "Bento, A. P.", "Chambers, J.", "Mendez, D.", "Mutowo, P.", "Atkinson, F.", "Bellis, L. J.", "Cibrian-Uhalte, E.", "Davies, M.", "Dedman, N.", "Karlsson, A.", "Magarinos, M. P.", "Overington, J. P.", "Papadatos, G.", "Smit, I.", "Leach, A. R."],
        "title": "The ChEMBL Database in 2017",
        "journal": "Nucleic Acids Research",
        "year": 2017,
        "volume": "45",
        "number": "D1",
        "pages": "D945--D954",
        "doi": "10.1093/nar/gkw1074",
    },
    {
        "key": "daina2019swisstarget",
        "type": "article",
        "authors": ["Daina, A.", "Michielin, O.", "Zoete, V."],
        "title": "SwissTargetPrediction: Updated Data and Its Application to the Identification of Drug Targets",
        "journal": "Nucleic Acids Research",
        "year": 2019,
        "volume": "47",
        "number": "W1",
        "pages": "W357--W364",
        "doi": "10.1093/nar/gkz382",
    },
    {
        "key": "gilson2016bindingdb",
        "type": "article",
        "authors": ["Gilson, M. K.", "Liu, T.", "Baitaluk, M.", "Nicola, G.", "Hwang, L.", "Chong, J."],
        "title": "BindingDB in 2015: A Public Database for Medicinal Chemistry, Computational Chemistry and Systems Pharmacology",
        "journal": "Nucleic Acids Research",
        "year": 2016,
        "volume": "44",
        "number": "D1",
        "pages": "D1045--D1053",
        "doi": "10.1093/nar/gkv1072",
    },
    {
        "key": "kim2023pubchem",
        "type": "article",
        "authors": ["Kim, S.", "Chen, J.", "Cheng, T.", "Gindulyte, A.", "He, J.", "He, S.", "Li, Q.", "Shoemaker, B. A.", "Thiessen, P. A.", "Yu, B.", "Zaslavsky, L.", "Zhang, J.", "Bolton, E. E."],
        "title": "PubChem 2023 Update",
        "journal": "Nucleic Acids Research",
        "year": 2023,
        "volume": "51",
        "number": "D1",
        "pages": "D1373--D1380",
        "doi": "10.1093/nar/gkac956",
    },
    {
        "key": "freshour2016dgidb",
        "type": "article",
        "authors": ["Freshour, S. L.", "Kietsiriroje, N.", "Cotto, K. C.", "Coppola, M. A.", "McMichael, J. F.", "Wagner, A. H.", "Griffith, O. L.", "Griffith, M."],
        "title": "DGIdb 3.0: A Redesign and Expansion of the Drug-Gene Interaction Database",
        "journal": "Nucleic Acids Research",
        "year": 2016,
        "volume": "44",
        "number": "D1",
        "pages": "D1050--D1053",
        "doi": "10.1093/nar/gkv1252",
    },
    {
        "key": "ochoa2023opentargets",
        "type": "article",
        "authors": ["Ochoa, D.", "Hercules, A.", "Carmona, M.", "Suveges, D.", "Gonzalez-Uriarte, A.", "Malangone, C.", "Miranda, A.", "Hill, R.", "Fumis, L.", "de la Vega, A.", "et al."],
        "title": "The Open Targets Platform: Supporting Systematic Drug Target Identification and Prioritisation",
        "journal": "Nucleic Acids Research",
        "year": 2023,
        "volume": "51",
        "number": "D1",
        "pages": "D1352--D1360",
        "doi": "10.1093/nar/gkac1046",
    },
    {
        "key": "berman2000pdb",
        "type": "article",
        "authors": ["Berman, H. M.", "Westbrook, J.", "Feng, Z.", "Gilliland, G.", "Bhat, T. N.", "Weissig, H.", "Shindyalov, I. N.", "Bourne, P. E."],
        "title": "The Protein Data Bank",
        "journal": "Nucleic Acids Research",
        "year": 2000,
        "volume": "28",
        "number": "1",
        "pages": "235--242",
        "doi": "10.1093/nar/28.1.235",
    },
    {
        "key": "szklarczyk2023string",
        "type": "article",
        "authors": ["Szklarczyk, D.", "Kirsch, R.", "Koutrouli, M.", "Nastou, K.", "Mehryary, F.", "Hachilif, R.", "Gable, A.", "Fang, T.", "Doncheva, N. T.", "Pyysalo, S.", "Bork, P.", "Jensen, L. J.", "von Mering, C."],
        "title": "The STRING Database in 2023: Protein-Protein Association Networks and Functional Enrichment Analyses for Any Sequenced Genome of Interest",
        "journal": "Nucleic Acids Research",
        "year": 2023,
        "volume": "51",
        "number": "D1",
        "pages": "D638--D646",
        "doi": "10.1093/nar/gkac1000",
    },
    {
        "key": "wishart2018drugbank",
        "type": "article",
        "authors": ["Wishart, D. S.", "Feunang, Y. D.", "Guo, A. C.", "Lo, E. J.", "Maran, A.", "Grant, J. R.", "Sajed, T.", "Johnson, D.", "Li, C.", "Sayeeda, Z.", "et al."],
        "title": "DrugBank 5.0: A Major Update to the DrugBank Database for 2018",
        "journal": "Nucleic Acids Research",
        "year": 2018,
        "volume": "46",
        "number": "D1",
        "pages": "D1074--D1082",
        "doi": "10.1093/nar/gkx1037",
    },
    {
        "key": "oboyle2011openbabel",
        "type": "article",
        "authors": ["O'Boyle, N. M.", "Banck, M.", "James, C. A.", "Morley, C.", "Vandermeersch, T.", "Hutchison, G. R."],
        "title": "Open Babel: An Open Chemical Toolbox",
        "journal": "Journal of Cheminformatics",
        "year": 2011,
        "volume": "3",
        "pages": "33",
        "doi": "10.1186/1758-2946-3-33",
    },
    {
        "key": "daina2016swissadme",
        "type": "article",
        "authors": ["Daina, A.", "Michielin, O.", "Zoete, V."],
        "title": "SwissADME: A Free Web Tool to Evaluate Pharmacokinetics, Drug-Likeness and Medicinal Chemistry Friendliness of Small Molecules",
        "journal": "Scientific Reports",
        "year": 2017,
        "volume": "7",
        "pages": "42717",
        "doi": "10.1038/srep42717",
    },
    {
        "key": "ertl2009sascore",
        "type": "article",
        "authors": ["Ertl, P.", "Schuffenhauer, A."],
        "title": "Estimation of Synthetic Accessibility Score of Drug-like Molecules Based on Molecular Complexity and Fragment Contributions",
        "journal": "Journal of Cheminformatics",
        "year": 2009,
        "volume": "1",
        "pages": "8",
        "doi": "10.1186/1758-2946-1-8",
    },
    {
        "key": "bickerton2012qed",
        "type": "article",
        "authors": ["Bickerton, G. R.", "Paolini, G. V.", "Besnard, J.", "Muresan, S.", "Hopkins, A. L."],
        "title": "Quantifying the Chemical Beauty of Drugs",
        "journal": "Nature Chemistry",
        "year": 2012,
        "volume": "4",
        "number": "2",
        "pages": "90--98",
        "doi": "10.1038/nchem.1243",
    },
    {
        "key": "lipinski1997rule",
        "type": "article",
        "authors": ["Lipinski, C. A.", "Lombardo, F.", "Dominy, B. W.", "Feeney, P. J."],
        "title": "Experimental and Computational Approaches to Estimate Solubility and Permeability in Drug Discovery and Development Settings",
        "journal": "Advanced Drug Delivery Reviews",
        "year": 1997,
        "volume": "23",
        "number": "1--3",
        "pages": "3--25",
        "doi": "10.1016/S0169-409X(96)00423-1",
    },
    {
        "key": "veber2002oral",
        "type": "article",
        "authors": ["Veber, D. F.", "Johnson, S. R.", "Cheng, H.-Y.", "Smith, B. R.", "Ward, K. W.", "Kopple, K. D."],
        "title": "Molecular Properties That Influence the Oral Bioavailability of Drug Candidates",
        "journal": "Journal of Medicinal Chemistry",
        "year": 2002,
        "volume": "45",
        "number": "12",
        "pages": "2615--2623",
        "doi": "10.1021/jm020017n",
    },
    {
        "key": "rdkit",
        "type": "misc",
        "authors": ["RDKit contributors"],
        "title": "RDKit: Open-Source Cheminformatics Software",
        "year": 2024,
        "url": "https://www.rdkit.org",
    },
]


def _bib_escape(value: str) -> str:
    return str(value).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def bibtex() -> str:
    """Return all references as one BibTeX file."""
    blocks = []
    for ref in REFERENCES:
        fields = [
            ("author", " and ".join(ref.get("authors") or [])),
            ("title", ref.get("title") or ""),
            ("journal", ref.get("journal") or ""),
            ("year", ref.get("year") or ""),
            ("volume", ref.get("volume") or ""),
            ("number", ref.get("number") or ""),
            ("pages", ref.get("pages") or ""),
            ("doi", ref.get("doi") or ""),
            ("url", ref.get("url") or ""),
        ]
        body = "\n".join(
            f"  {name} = {{{_bib_escape(value)}}},"
            for name, value in fields
            if value not in (None, "")
        )
        blocks.append(f"@{ref.get('type', 'article')}{{{ref['key']},\n{body}\n}}")
    header = "% DrugHX curated bibliography\n"
    return header + "\n\n".join(blocks) + "\n"


def ris() -> str:
    """Return all references as one RIS file."""
    records = []
    for ref in REFERENCES:
        lines = ["TY  - " + ("JOUR" if ref.get("type") == "article" else "GEN")]
        for author in ref.get("authors") or []:
            lines.append(f"AU  - {author}")
        lines.append(f"TI  - {ref.get('title') or ''}")
        if ref.get("journal"):
            lines.append(f"JO  - {ref['journal']}")
        if ref.get("year"):
            lines.append(f"PY  - {ref['year']}")
        if ref.get("volume"):
            lines.append(f"VL  - {ref['volume']}")
        if ref.get("number"):
            lines.append(f"IS  - {ref['number']}")
        pages = str(ref.get("pages") or "")
        if pages:
            start = pages.split("--")[0]
            end = pages.split("--")[-1]
            lines.append(f"SP  - {start}")
            if end and end != start:
                lines.append(f"EP  - {end}")
        if ref.get("doi"):
            lines.append(f"DO  - {ref['doi']}")
        if ref.get("url"):
            lines.append(f"UR  - {ref['url']}")
        lines.append("ER  - ")
        records.append("\n".join(lines))
    return "\n\n".join(records) + "\n"


def count() -> int:
    return len(REFERENCES)
