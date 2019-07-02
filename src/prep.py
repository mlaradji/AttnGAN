"""Preparation"""

from pathlib import Path
import urllib.request
import tarfile

from google_drive_downloader import GoogleDriveDownloader as gdd

def main():
    # Define data directories.
    folder = dict()
    folder["base"] = Path("..")
    folder["data"] = folder["base"] / "data"
    folder["models"] = folder["base"] / "models"
    folder["damsme"] = folder["base"] / "DAMSMencoders"

    for d in folder:
        folder[d].mkdir(parents=True, exist_ok=True)

    # Download COCO dataset
    downloads = [
        # Preprocessed metadata
        {"source": "gdrive", "id": "1O_LtUP9sch09QH3s_EBAgLEctBQ5JBSJ", "path": folder["data"] / "birds.zip"},
        {"source": "gdrive", "id": "1rSnbIGNDGZeHlsUlLdahj0RJ9oo6lgH9", "path": folder["data"] / "coco.zip"},
        # DAMSMencoders
        {"source": "gdrive", "id": "1GNUKjVeyWYBJ8hEU-yrfYQpDOkxEyP3V", "path": folder["damsme"] / "bird.zip"},
        {"source": "gdrive", "id": "1zIrXCE9F6yfbEJIbNP5-YrEe2pZcPSGJ", "path": folder["damsme"] / "coco.zip"},
        # Models
        {
            "source": "gdrive", "id": "1lqNG75suOuR_8gjoEPYNp8VyT_ufPPig",
            "path": folder["models"] / "bird_AttnGAN2.pth",
        },
        {
            "source": "gdrive", "id": "1i9Xkg9nU74RAvkcqKE-rJYhjvzKAMnCi",
            "path": folder["models"] / "coco_AttnGAN2.pth",
        },
        {
            "source": "gdrive", "id": "19TG0JUoXurxsmZLaJ82Yo6O0UJ6aDBpg",
            "path": folder["models"] / "bird_AttnDCGAN2.pth",
        },
        # Image data.
        {"source": "url", "url": "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz", "path": folder["data"] / "CUB_200_2011.tgz", "extract_to": folder["data"]/"birds"},
    ]

    for file in downloads:

        if file["path"].exists():
            print("Skipped downloading '%s' as it already exists." % file["path"])
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
    if ''.join(file["path"].suffixes) in {".tgz", ".tar.gz"}:
        tar = tarfile.open(path)
        tar.extractall(path=extract_to)
        tar.close()

if __name__ == "__main__":
    main()
