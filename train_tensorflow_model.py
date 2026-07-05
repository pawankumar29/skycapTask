from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_AUTHENTICITY_THRESHOLDS = {
    "fake_reject_probability": 0.75,
    "fake_reject_margin": 0.20,
    "fake_review_probability": 0.75,
    "fake_review_margin": 0.10,
}


def build_model(image_size: tuple[int, int], class_count: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(image_size[0], image_size[1], 3))
    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)
    x = tf.keras.layers.RandomFlip("horizontal")(x)
    x = tf.keras.layers.RandomRotation(0.05)(x)
    x = tf.keras.layers.RandomZoom(0.08)(x)
    x = tf.keras.layers.RandomTranslation(0.04, 0.04)(x)
    x = tf.keras.layers.RandomContrast(0.12)(x)

    for filters in (32, 64, 128, 192):
        x = tf.keras.layers.SeparableConv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.SeparableConv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.SpatialDropout2D(0.08)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(160, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def class_weights(dataset: tf.data.Dataset) -> dict[int, float]:
    counts: dict[int, int] = {}
    for _, labels in dataset.unbatch():
        label = int(labels.numpy())
        counts[label] = counts.get(label, 0) + 1
    total = sum(counts.values())
    classes = max(1, len(counts))
    return {label: total / (classes * count) for label, count in counts.items()}


def validation_report(model: tf.keras.Model, dataset: tf.data.Dataset, class_names: list[str]) -> dict:
    predictions = model.predict(dataset, verbose=0)
    predicted_labels = predictions.argmax(axis=1)
    true_labels = []
    for _, labels in dataset:
        true_labels.extend(int(label) for label in labels.numpy())

    fake_index = class_names.index("fake")
    real_index = class_names.index("real")
    false_accepts = 0
    false_rejects = 0
    correct = 0
    confusion = {
        "fake_as_fake": 0,
        "fake_as_real": 0,
        "real_as_real": 0,
        "real_as_fake": 0,
    }

    for true_label, predicted_label in zip(true_labels, predicted_labels):
        correct += int(true_label == predicted_label)
        if true_label == fake_index and predicted_label == fake_index:
            confusion["fake_as_fake"] += 1
        elif true_label == fake_index and predicted_label == real_index:
            false_accepts += 1
            confusion["fake_as_real"] += 1
        elif true_label == real_index and predicted_label == real_index:
            confusion["real_as_real"] += 1
        elif true_label == real_index and predicted_label == fake_index:
            false_rejects += 1
            confusion["real_as_fake"] += 1

    total = max(1, len(true_labels))
    return {
        "accuracy": correct / total,
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
        "confusion": confusion,
        "samples": len(true_labels),
    }


def generic_validation_report(model: tf.keras.Model, dataset: tf.data.Dataset, class_names: list[str]) -> dict:
    predictions = model.predict(dataset, verbose=0)
    predicted_labels = predictions.argmax(axis=1)
    true_labels = []
    for _, labels in dataset:
        true_labels.extend(int(label) for label in labels.numpy())

    correct = sum(int(true_label == predicted_label) for true_label, predicted_label in zip(true_labels, predicted_labels))
    confusion = {class_name: {name: 0 for name in class_names} for class_name in class_names}
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        confusion[class_names[true_label]][class_names[predicted_label]] += 1

    total = max(1, len(true_labels))
    return {
        "accuracy": correct / total,
        "confusion": confusion,
        "samples": len(true_labels),
    }


def train_classifier(
    data_dir: Path,
    model_path: Path,
    metadata_path: Path,
    image_size: tuple[int, int],
    batch_size: int,
    epochs: int,
    prefetch: int,
    required_classes: set[str] | None = None,
    thresholds: dict | None = None,
    validation_kind: str = "generic",
) -> None:
    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )

    class_names = list(train_ds.class_names)
    if required_classes:
        missing_classes = required_classes - set(class_names)
        if missing_classes:
            raise SystemExit(f"Expected class folders. Missing: {', '.join(sorted(missing_classes))}")

    weights = class_weights(train_ds)
    train_ds = train_ds.prefetch(prefetch)
    val_ds = val_ds.prefetch(prefetch)

    model = build_model(image_size, len(class_names))
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(model_path), monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=weights, callbacks=callbacks)
    report = (
        validation_report(model, val_ds, class_names)
        if validation_kind == "authenticity"
        else generic_validation_report(model, val_ds, class_names)
    )

    model.save(str(model_path))
    metadata = {
        "class_names": class_names,
        "image_size": [image_size[0], image_size[1]],
        "validation": report,
    }
    if thresholds:
        metadata["thresholds"] = thresholds
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def has_required_class_images(data_dir: Path, required_classes: set[str]) -> tuple[bool, list[str]]:
    if not data_dir.exists():
        return False, [f"missing directory: {data_dir}"]

    problems = []
    for class_name in sorted(required_classes):
        class_dir = data_dir / class_name
        if not class_dir.is_dir():
            problems.append(f"missing folder: {class_dir}")
            continue
        has_images = any(path.suffix.lower() in IMAGE_EXTENSIONS for path in class_dir.rglob("*") if path.is_file())
        if not has_images:
            problems.append(f"no images in: {class_dir}")

    return not problems, problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TensorFlow currency classifiers.")
    parser.add_argument("--data", default="currency/data", help="Directory containing fake/ and real/ folders.")
    parser.add_argument("--model", default="currency_authenticity_model.keras", help="Output Keras model path.")
    parser.add_argument("--metadata", default="currency_authenticity_model.json", help="Output metadata JSON path.")
    parser.add_argument("--denomination-data", default="currency/denomination_data", help="Directory containing denomination class folders.")
    parser.add_argument("--denomination-model", default="currency_denomination_model.keras", help="Output denomination Keras model path.")
    parser.add_argument("--denomination-metadata", default="currency_denomination_model.json", help="Output denomination metadata JSON path.")
    parser.add_argument("--skip-denomination", action="store_true", help="Train only the real/fake authenticity model.")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--denomination-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--prefetch", type=int, default=1)
    parser.add_argument("--fake-reject-probability", type=float, default=DEFAULT_AUTHENTICITY_THRESHOLDS["fake_reject_probability"])
    parser.add_argument("--fake-reject-margin", type=float, default=DEFAULT_AUTHENTICITY_THRESHOLDS["fake_reject_margin"])
    parser.add_argument("--fake-review-probability", type=float, default=DEFAULT_AUTHENTICITY_THRESHOLDS["fake_review_probability"])
    parser.add_argument("--fake-review-margin", type=float, default=DEFAULT_AUTHENTICITY_THRESHOLDS["fake_review_margin"])
    args = parser.parse_args()

    image_size = (args.image_size, args.image_size)
    thresholds = {
        "fake_reject_probability": args.fake_reject_probability,
        "fake_reject_margin": args.fake_reject_margin,
        "fake_review_probability": args.fake_review_probability,
        "fake_review_margin": args.fake_review_margin,
    }

    train_classifier(
        data_dir=Path(args.data),
        model_path=Path(args.model),
        metadata_path=Path(args.metadata),
        image_size=image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        prefetch=args.prefetch,
        required_classes={"fake", "real"},
        thresholds=thresholds,
        validation_kind="authenticity",
    )

    denomination_classes = {"10", "20", "50", "100", "200", "500", "2000"}
    denomination_ready, denomination_problems = has_required_class_images(Path(args.denomination_data), denomination_classes)
    if args.skip_denomination:
        print("Skipping denomination training because --skip-denomination was provided.")
    elif not denomination_ready:
        print("Skipping denomination training because the denomination dataset is incomplete:")
        for problem in denomination_problems:
            print(f"- {problem}")
    else:
        train_classifier(
            data_dir=Path(args.denomination_data),
            model_path=Path(args.denomination_model),
            metadata_path=Path(args.denomination_metadata),
            image_size=image_size,
            batch_size=args.batch_size,
            epochs=args.denomination_epochs or args.epochs,
            prefetch=args.prefetch,
            required_classes=denomination_classes,
            validation_kind="denomination",
        )


if __name__ == "__main__":
    main()
