Default **no** MD:

```
python scripts/03_reconstruct_side_chains.py outputs/my_protein
```
can also use a custom output dir like so: `--output-dir outputs/reconstructed`

NVT MD for **all** frames:

```
python scripts/03_reconstruct_side_chains.py outputs/my_protein --md
```

MD only for **selected** 0-based frames (0 = first):

```
python scripts/03_reconstruct_side_chains.py outputs/my_protein \
  --md \
  --md-frames 0 5 10-14
```