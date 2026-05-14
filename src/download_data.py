import argparse
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

from config import DEFAULT_RAW_FILENAME, RAW_DATA_DIR


def build_parser():
    parser = argparse.ArgumentParser(
        description="Download or copy raw trajectory data into the local data directory."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--url", help="HTTP(S) URL of the raw dataset file or archive")
    input_group.add_argument("--source", type=Path, help="Local raw dataset file or archive to copy")
    parser.add_argument("--output_dir", type=Path, default=RAW_DATA_DIR, help="Directory for downloaded raw files")
    parser.add_argument(
        "--filename",
        default=None,
        help=f"Output filename. Defaults to the URL/source basename, or {DEFAULT_RAW_FILENAME!r} if it cannot be inferred.",
    )
    parser.add_argument("--extract", action="store_true", help="Extract .zip/.tar/.tar.gz/.tgz archives after download/copy")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output file")
    return parser


def infer_filename(url=None, source=None, explicit=None):
    if explicit:
        return explicit
    if source:
        return source.name
    parsed = urlparse(url)
    name = Path(unquote(parsed.path)).name
    return name or DEFAULT_RAW_FILENAME


def copy_local_file(source, destination, overwrite=False):
    if not source.exists():
        raise FileNotFoundError(f"本地文件不存在: {source}")
    if destination.exists() and not overwrite:
        print(f"文件已存在，跳过复制: {destination}")
        return destination
    shutil.copy2(source, destination)
    return destination


def download_file(url, destination, overwrite=False):
    if destination.exists() and not overwrite:
        print(f"文件已存在，跳过下载: {destination}")
        return destination

    def reporthook(block_count, block_size, total_size):
        downloaded = block_count * block_size
        if total_size > 0:
            percent = min(downloaded / total_size * 100, 100)
            print(f"\rDownloading: {percent:6.2f}% ({downloaded}/{total_size} bytes)", end="")
        else:
            print(f"\rDownloading: {downloaded} bytes", end="")

    urllib.request.urlretrieve(url, destination, reporthook=reporthook)
    print()
    return destination


def extract_archive(path, output_dir):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            archive.extractall(output_dir)
        print(f"已解压 zip 到: {output_dir}")
        return

    if tarfile.is_tarfile(path):
        with tarfile.open(path) as archive:
            archive.extractall(output_dir)
        print(f"已解压 tar 到: {output_dir}")
        return

    print(f"不是可识别的压缩包，跳过解压: {path}")


def main():
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    filename = infer_filename(args.url, args.source, args.filename)
    destination = args.output_dir / filename

    if args.source:
        print(f"Copying raw data from: {args.source}")
        output_path = copy_local_file(args.source, destination, args.overwrite)
    else:
        print(f"Downloading raw data from: {args.url}")
        output_path = download_file(args.url, destination, args.overwrite)

    print(f"Raw data saved to: {output_path}")

    if args.extract:
        extract_archive(output_path, args.output_dir)


if __name__ == "__main__":
    main()
