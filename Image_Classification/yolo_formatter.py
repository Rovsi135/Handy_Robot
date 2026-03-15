import os
import random
import shutil
import argparse


def create_yolo_split(labels_dir, images_dir, output_dir, train_ratio=0.85, seed=42):
    labels_dir = os.path.abspath(labels_dir)
    images_dir = os.path.abspath(images_dir)
    output_dir = os.path.abspath(output_dir)

    # Output folders in YOLO format
    train_images_dir = os.path.join(output_dir, "images", "train")
    val_images_dir = os.path.join(output_dir, "images", "val")
    train_labels_dir = os.path.join(output_dir, "labels", "train")
    val_labels_dir = os.path.join(output_dir, "labels", "val")

    for folder in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
        os.makedirs(folder, exist_ok=True)

    # Get all label files
    label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]

    matched_pairs = []
    missing_images = []

    for label_file in label_files:
        base_name = os.path.splitext(label_file)[0]
        image_file = base_name + ".jpg"

        image_path = os.path.join(images_dir, image_file)
        label_path = os.path.join(labels_dir, label_file)

        if os.path.isfile(image_path):
            matched_pairs.append((image_path, label_path, image_file, label_file))
        else:
            missing_images.append(image_file)

    if not matched_pairs:
        print("No matching labeled images were found.")
        return

    # Shuffle randomly
    random.seed(seed)
    random.shuffle(matched_pairs)

    # Split
    split_index = int(len(matched_pairs) * train_ratio)
    train_pairs = matched_pairs[:split_index]
    val_pairs = matched_pairs[split_index:]

    # Copy files
    for image_path, label_path, image_file, label_file in train_pairs:
        shutil.copy2(image_path, os.path.join(train_images_dir, image_file))
        shutil.copy2(label_path, os.path.join(train_labels_dir, label_file))

    for image_path, label_path, image_file, label_file in val_pairs:
        shutil.copy2(image_path, os.path.join(val_images_dir, image_file))
        shutil.copy2(label_path, os.path.join(val_labels_dir, label_file))

    # Summary
    print(f"Total label files found: {len(label_files)}")
    print(f"Matched labeled images: {len(matched_pairs)}")
    print(f"Training set: {len(train_pairs)}")
    print(f"Validation set: {len(val_pairs)}")

    if missing_images:
        print(f"\nWarning: {len(missing_images)} labels had no matching .jpg image.")
        print("Examples:")
        for name in missing_images[:10]:
            print(f"  - {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create YOLO train/val split using only already-labeled images.")
    parser.add_argument("labels_dir", help="Path to folder containing label .txt files")
    parser.add_argument("images_dir", help="Path to folder containing .jpg images")
    parser.add_argument("output_dir", help="Path to output YOLO dataset folder")
    parser.add_argument("--train_ratio", type=float, default=0.85, help="Train split ratio (default: 0.85)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    create_yolo_split(
        labels_dir=args.labels_dir,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed
    )