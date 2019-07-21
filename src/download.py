"""Preparation"""

from typing import Dict, Tuple, Set
from pathlib import Path
import argparse
import tarfile
import urllib.request

from google_drive_downloader import GoogleDriveDownloader as gdd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download data, preprocessed metadata, pretrained DAMSencoders and fully trained models for either the birds or the COCO dataset."
    )
    parser.add_argument(
        "dataset",
        type=str,
        default="birds",
        help="Choose the dataset to download. Available options are 'birds' and 'coco'.",
    )
    parser.add_argument(
        "--all",
        action="store_const",
        default=False,
        const=True,
        help="Download everything. Further subsetting arguments are ignored.",
    )
    parser.add_argument(
        "--raw",
        action="store_const",
        default=False,
        const=True,
        help="Download the raw images for the dataset.",
    )
    parser.add_argument(
        "--metadata",
        action="store_const",
        default=False,
        const=True,
        help="Download preprocessed metadata.",
    )
    parser.add_argument(
        "--damsme",
        action="store_const",
        default=False,
        const=True,
        help="Download pretrained DAMSMencoders.",
    )
    parser.add_argument(
        "--models",
        action="store_const",
        default=False,
        const=True,
        help="Download fully-trained models.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_const",
        default=False,
        const=True,
        help="Overwrite downloaded/extracted files if they already exist.",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="The path of the parent directory of the 'data', 'models' and 'DAMSMencoders' directories in which the files will be downloded. The path should be relative to the root of AttnGAN.",
    )
    args = parser.parse_args()
    return args


def main(args):
    # Define data directories.
    folder = dict()
    folder["base"] = Path(args.base_dir)
    folder["data"] = folder["base"] / "data"
    folder["models"] = folder["base"] / "models"
    folder["damsme"] = folder["base"] / "DAMSMencoders"

    for d in folder:
        folder[d].mkdir(parents=True, exist_ok=True)

    downloads = {
        "birds": [
            # Preprocessed metadata
            {
                "source": "gdrive",
                "id": "1O_LtUP9sch09QH3s_EBAgLEctBQ5JBSJ",
                "path": folder["data"] / "birds.zip",
                "type": "metadata",
            },
            # DAMSMencoders
            {
                "source": "gdrive",
                "id": "1GNUKjVeyWYBJ8hEU-yrfYQpDOkxEyP3V",
                "path": folder["damsme"] / "bird.zip",
                "type": "damsme",
            },
            # Models
            {
                "source": "gdrive",
                "id": "1lqNG75suOuR_8gjoEPYNp8VyT_ufPPig",
                "path": folder["models"] / "bird_AttnGAN2.pth",
                "type": "models",
            },
            {
                "source": "gdrive",
                "id": "19TG0JUoXurxsmZLaJ82Yo6O0UJ6aDBpg",
                "path": folder["models"] / "bird_AttnDCGAN2.pth",
                "type": "models",
            },
            # Image data.
            {
                "source": "url",
                "url": "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz",
                "path": folder["data"] / "CUB_200_2011.tgz",
                "extract_to": folder["data"] / "birds",
                "type": "raw",
            },
        ],
        "coco": [
            # Preprocessed metadata
            {
                "source": "gdrive",
                "id": "1rSnbIGNDGZeHlsUlLdahj0RJ9oo6lgH9",
                "path": folder["data"] / "coco.zip",
                "type": "metadata",
            },
            # DAMSMencoders
            {
                "source": "gdrive",
                "id": "1zIrXCE9F6yfbEJIbNP5-YrEe2pZcPSGJ",
                "path": folder["damsme"] / "coco.zip",
                "type": "damsme",
            },
            # Models
            {
                "source": "gdrive",
                "id": "1i9Xkg9nU74RAvkcqKE-rJYhjvzKAMnCi",
                "path": folder["models"] / "coco_AttnGAN2.pth",
                "type": "models",
            },
            # Image data.
        ],
    }

    allowed_types = conditional_set(
        *(
            ("metadata", args.metadata),
            ("damsme", args.damsme),
            ("raw", args.raw),
            ("models", args.models),
        )
    )
    print(allowed_types)
    for file in filter(lambda d: d["type"] in allowed_types, downloads[args.dataset]):

        if not (args.overwrite) and file["path"].exists():
            print(
                "Skipped downloading '%s' as it already exists. Use the '--overwrite' flag to force redownload."
                % file["path"]
            )
            continue

        if file["source"] == "gdrive":
            gdd.download_file_from_google_drive(
                file_id=file["id"],
                dest_path=file["path"],
                unzip=file["path"].suffix == ".zip",
            )
        elif file["source"] == "url":
            download_file(url=file["url"], save_path=file["path"])
            extract_file(path=file["path"], extract_to=file["extract_to"])

    # # Choose the dataset.
    # coco_dataset = "train2014"
    # while remaining_tries > 0:
    #     try:
    #         (data_dir / "coco" / "text").symlink_to(
    #             data_dir / "coco" / coco_dataset, target_is_directory=True
    #         )
    #         break
    #     except FileExistsError:
    #         (data_dir / "coco" / "text").unlink()
    #         continue


def download_file(url: str, save_path: str) -> None:
    print(f"Downloading {url} into {save_path}...")

    urllib.request.urlretrieve(url, save_path)

    # response = requests.get(url, stream=True)

    # with open(save_path, "wb") as handle:
    #     for data in tqdm(response.iter_content()):
    #         handle.write(data)

    print(f"Successfully downloaded {url} into {save_path}.")


def extract_file(path: str, extract_to: str = None) -> None:
    if "".join(path.suffixes) in {".tgz", ".tar.gz"}:
        tar = tarfile.open(path)
        tar.extractall(path=extract_to)
        tar.close()


def conditional_set(*sources: Tuple[object, bool]) -> Set[object]:
    output = set()
    for source in sources:
        if source[1]:
            output.add(source[0])

    return output


if __name__ == "__main__":
    args = parse_args()

    if args.all:
        args.damsme = True
        args.metadata = True
        args.raw = True
        args.models = True

    main(args)
