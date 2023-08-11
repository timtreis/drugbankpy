import warnings
import zipfile

import pandas
import pandas as pd
from fuzzywuzzy import fuzz
from lxml import etree
from nltk.metrics.distance import edit_distance


class DrugBankConnector:
    """
    Connector for DrugBank data.

    Attributes
    ----------
    - file: File path to the DrugBank XML or ZIP.
    - root: Root element of the parsed XML data.
    - drugs: DataFrame holding the processed DrugBank data.
    """

    def __init__(self, file: str):
        """Initialize the DrugBankConnector.

        Args:
        - file (str): Path to the XML or ZIP file containing DrugBank data.
        """
        self.file = file
        self.root = None
        self.drugs = None

    def _initialize(self):
        """Initialize the XML root element from the provided file."""
        if self.root is None:
            if self.file.endswith(".xml"):
                with open(self.file, "rb") as f:
                    tree = etree.parse(f)
            elif self.file.endswith(".zip"):
                with zipfile.ZipFile(self.file, "r") as zf:
                    for name in zf.namelist():
                        if name.endswith(".xml"):
                            with zf.open(name) as f:
                                tree = etree.parse(f)
                                break
                    else:
                        raise Exception("No XML file found in the zip")
            else:
                raise Exception("File must be either an .xml or .zip file")

            self.root = tree.getroot()

    def load_drugs(self, return_df: bool = False) -> pd.DataFrame:
        """Load drugs data from the XML file into a DataFrame.

        Args:
        - return_df (bool, optional): Whether to return the loaded data as a DataFrame. Defaults to False.

        Returns
        -------
        - pd.DataFrame: DataFrame of loaded drugs data if return_df is True.
        """
        self._initialize()

        ns = "{http://www.drugbank.ca}"
        rows = []
        for drug in self.root:
            if drug.tag != f"{ns}drug":
                raise ValueError(f"Unexpected tag: Expected '{ns}drug'")

            # Define the required xpath expressions for data extraction
            aliases_xpath = ".//db:international-brands/db:international-brand | .//db:synonyms/db:synonym[@language='English'] | .//db:products/db:product/db:name"
            aliases = {elem.text for elem in drug.xpath(aliases_xpath, namespaces={"db": "http://www.drugbank.ca"})}

            aliases.add(drug.findtext(f"{ns}name"))
            row = {
                "type": drug.get("type"),
                "drugbank_id": drug.findtext(f"{ns}drugbank-id[@primary='true']"),
                "name": drug.findtext(f"{ns}name"),
                "description": drug.findtext(f"{ns}description"),
                "groups": "|".join([group.text for group in drug.findall(f"{ns}groups/{ns}group")]),
                "atc_codes": "|".join([code.get("code") for code in drug.findall(f"{ns}atc-codes/{ns}atc-code")]),
                "categories": "|".join(
                    [x.findtext(f"{ns}category") for x in drug.findall(f"{ns}categories/{ns}category")]
                ),
                "inchi": drug.findtext(f"{ns}calculated-properties/{ns}property[{ns}kind='InChI']/{ns}value"),
                "inchikey": drug.findtext(f"{ns}calculated-properties/{ns}property[{ns}kind='InChIKey']/{ns}value"),
                "aliases": "|".join(sorted(aliases)),
            }
            rows.append(row)

        columns = [
            "drugbank_id",
            "name",
            "type",
            "groups",
            "atc_codes",
            "categories",
            "inchikey",
            "inchi",
            "description",
            "aliases",
        ]
        drugs = pandas.DataFrame(rows, columns=columns)

        # Split aliases into separate rows
        drugs = drugs.assign(aliases=drugs.aliases.str.split("|")).explode("aliases")
        drugs = drugs.rename(columns={"name": "primary_name", "aliases": "alias_name"})
        cols = ["drugbank_id", "primary_name", "alias_name"] + [
            col for col in drugs.columns if col not in ["drugbank_id", "primary_name", "alias_name"]
        ]
        drugs = drugs[cols]

        drugs.reset_index(drop=True, inplace=True)

        self.drugs = drugs

        if return_df:
            return drugs

    def find_drug(
        self, drug_name: str, exact: bool = True, fuzzy_threshold: int = 90, return_best: bool or int = False
    ) -> pd.DataFrame or None:
        """Find a drug by name.

        Args:
        - drug_name (str): The name of the drug to be searched.
        - exact (bool, optional): Whether to search for an exact match. Defaults to True.
        - fuzzy_threshold (int, optional): The minimum fuzzy match score for considering a match. Used if exact is False. Defaults to 90.
        - return_best (bool or int, optional): If True, return only the best match. If an integer n, return the top n matches. If False, return all matches.

        Returns
        -------
        - pd.DataFrame or None: DataFrame with the search results, or None if not found.
        """
        if not hasattr(self, "drugs") or self.drugs is None:
            print("First time using connector, loading drugs...")
            self.load_drugs()

        if exact:
            result = self.drugs.query("`alias_name` == @drug_name")
        else:
            drug_name = drug_name.lower().replace(" ", "")
            self.drugs["clean_alias_name"] = self.drugs["alias_name"].str.lower().str.replace(r" ", "")
            self.drugs["fuzzy_score"] = self.drugs.apply(
                lambda row: fuzz.token_set_ratio(row["clean_alias_name"], drug_name), axis=1
            )
            result = self.drugs[self.drugs["fuzzy_score"] >= fuzzy_threshold]

            if len(result) == 0:
                warnings.warn(f"Drug {drug_name} not found.", stacklevel=2)
                return None

            result["edit_distance"] = result.apply(
                lambda row: edit_distance(row["clean_alias_name"], drug_name), axis=1
            )
            result = result.sort_values("fuzzy_score", ascending=False)

        if isinstance(return_best, bool) and return_best:
            return result.head(1)
        elif isinstance(return_best, bool) and not return_best:
            return result
        elif isinstance(return_best, int):
            return result.head(return_best)
