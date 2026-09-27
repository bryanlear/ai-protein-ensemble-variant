import argparse
from pathlib import Path
from urllib.request import urlretrieve

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("uniprot_id", help="UniProt ID of protein to fetch")
    args = parser.parse_args()
    uniprot_id = args.uniprot_id.upper()

    directory = Path("data") / uniprot_id
    directory.mkdir(parents=True, exist_ok=True)
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    urlretrieve(url, directory / f"{uniprot_id}.fasta")

if __name__ == "__main__":
    main()

#from root --> python3 scripts/01_fetch_protein_sequence.py [UniProt ID]