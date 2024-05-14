import os
import logging
import argparse
from huggingface_hub import HfFileSystem


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--repo-id',
        type=str,
        required=True,
        help='HuggingFace repository ID',
    )
    parser.add_argument(
        '--path-in-repo',
        type=str,
        required=True,
        help='Path to download in the repository',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing files',
    )
    parser.add_argument(
        '--repo-type',
        type=str,
        default='dataset',
        help='Type of the repository',
    )
    return parser.parse_args()


def get_paths(path: str) -> list:
    '''
    Get all paths in the HuggingFace repository.

    Parameters
    ----------
    path : str
        Path to the repository.

    Returns
    -------
    list
        List of paths in the repository.
    '''
    fs = HfFileSystem()
    if fs.isfile(path):
        return [(path, None)]
    hf_paths = []
    for root, _, files in fs.walk(path):
        for file in files:
            hf_paths.append((root, file))
    return hf_paths


def download_from_hf(
    repo_id: str,
    path_in_repo: str,
    output_dir: str,
    overwrite: bool = False,
    repo_type: str = 'dataset',
):
    '''
    Download files from HuggingFace repository.

    Parameters
    ----------
    repo_id : str
        HuggingFace repository ID.
    path_in_repo : str
        Path to download in the repository.
    output_dir : str
        Output directory.
    overwrite : bool, False
        Overwrite existing files.
    repo_type : str, 'dataset'
        Type of the repository.
    '''
    if repo_type == 'dataset':
        hf_root_path = f'datasets/{repo_id}/{path_in_repo}'
    elif repo_type == 'model':
        hf_root_path = f'{repo_id}/{path_in_repo}'
    else:
        logging.error("repo_type should be either 'dataset' or 'model'")
        return

    fs = HfFileSystem()
    os.makedirs(output_dir, exist_ok=True)
    hf_paths = get_paths(hf_root_path)
    hf_root_path = os.path.normpath(hf_root_path)
    for i, (root, file) in enumerate(hf_paths):
        if file is None:
            file_output_dir = output_dir
            file = os.path.basename(root)
            root = os.path.dirname(root)
        else:
            file_output_dir = os.path.join(
                output_dir,
                os.path.relpath(root, os.path.dirname(hf_root_path)),
            )
        logging.info(f'[{i + 1}/{len(hf_paths)}] Processing {root}/{file}')

        if not os.path.exists(file_output_dir):
            os.makedirs(file_output_dir, exist_ok=True)

        if overwrite or not os.path.exists(os.path.join(file_output_dir, file)):
            fs.download(
                f'{root}/{file}',
                file_output_dir,
                verbose=False,
            )
            logging.info(f'\tDownloaded to {file_output_dir}')
        else:
            logging.info(f'\tFile exists in {file_output_dir}')


def main(args: argparse.Namespace):
    '''
    Main function to download files from HuggingFace repository.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments.
    '''
    download_from_hf(
        repo_id=args.repo_id,
        path_in_repo=args.path_in_repo,
        output_dir=os.path.normpath(args.output_dir),
        overwrite=args.overwrite,
        repo_type=args.repo_type,
    )


if __name__ == '__main__':
    logging.info('Downloading files from HuggingFace Hub')
    logging.info('If your data is private, please login to Hugging or set it public')
    args = get_args()
    logging.info(f'Write mode: {"overwrite" if args.overwrite else "skip"}')
    main(args=args)
    logging.info('Download completed')
