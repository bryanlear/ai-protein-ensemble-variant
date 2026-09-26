import argparse
import json
import re
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen, urlretrieve

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("uniprot_id", help="UniProt ID of protein to fetch")
    args = parser.parse_args()
    uniprot_id = args.uniprot_id.upper()

    if not re.fullmatch(r"[A-Z0-9-]+", uniprot_id):
        parser.error("invalid UniProt ID")

    directory = Path("data") / uniprot_id
    directory.mkdir(parents=True, exist_ok=True)

    base_url = "https://alphafold.ebi.ac.uk/api"

    with urlopen(f"{base_url}/prediction/{uniprot_id}") as response:
        prediction = json.load(response)[0]
    urlretrieve(prediction["pdbUrl"], directory / f"{uniprot_id}.pdb")

    try:
        with urlopen(f"{base_url}/complex/{uniprot_id}") as response:
            complexes = json.load(response)
    except HTTPError as error:
        if error.code != 404:
            raise
        complexes = []
    (directory / "complexes.json").write_text(json.dumps(complexes, indent=2) + "\n")

if __name__ == "__main__":
    main()
