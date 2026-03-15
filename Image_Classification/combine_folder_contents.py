from pathlib import Path
import shutil
import argparse

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect_images(input_folders: list[Path], output_folder: Path) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)

    for folder_index, folder in enumerate(input_folders, start=1):
        if not folder.exists() or not folder.is_dir():
            print(f"[Skip] Not a valid folder: {folder}")
            continue

        for image_path in sorted(folder.iterdir()):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            new_name = f"{folder_index}_{image_path.name}"
            destination = output_folder / new_name

            # In case even prefixed names still collide somehow
            counter = 1
            while destination.exists():
                destination = output_folder / f"{folder_index}_{counter}_{image_path.name}"
                counter += 1

            shutil.copy2(image_path, destination)
            print(f"[Copied] {image_path} -> {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge images from multiple folders into one folder with prefixed filenames."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="List of input folders containing images",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output folder where all renamed images will be copied",
    )
    args = parser.parse_args()

    input_folders = [Path(folder).expanduser().resolve() for folder in args.inputs]
    output_folder = Path(args.output).expanduser().resolve()

    collect_images(input_folders, output_folder)


if __name__ == "__main__":
    main()