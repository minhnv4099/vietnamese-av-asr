# Downloading from HuggingFace
To download a file or folder from a HuggingFace repo, use the following script:
```
python tools/download_from_HF.py \
    --repo-id huggingface/repo-id \
    --path-in-repo path/to/file/or/folder/in/repo \
    --output-dir path/to/directory/to/save/downloaded/data
```
If your repo is private, please login or set it to public before running this script.