"""
Fine-tune MobileNetV2 for binary fire classification.

Two-phase training:
  Phase 1 — Train only the new classification head (backbone frozen).
             Fast, low risk of overfitting, good starting weights.
  Phase 2 — Unfreeze the top layers of the backbone and fine-tune end-to-end
             at a low learning rate. Squeezes out extra accuracy.

Usage
-----
    python src/train_mobilenet.py --frames frames/ --epochs-head 10 --epochs-finetune 10
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import argparse

def train(frames_dir: str, model_out: str, epochs_head: int, epochs_finetune: int):
    import tensorflow as tf
    import numpy as np

    print(f"TensorFlow version: {tf.__version__}")
    print(f"Loading images from: {frames_dir}")

    # ------------------------------------------------------------------
    # 1. Load dataset with 80/20 train/val split
    # ------------------------------------------------------------------
    IMG_SIZE   = (224, 224)
    BATCH_SIZE = 32

    train_ds = tf.keras.utils.image_dataset_from_directory(
        frames_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        validation_split=0.2,
        subset="training",
        seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        frames_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        validation_split=0.2,
        subset="validation",
        seed=42,
    )

    # Cache and prefetch for speed
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(AUTOTUNE)

    # ------------------------------------------------------------------
    # 2. Data augmentation (applied during training only)
    #    Helps generalise to unseen fire angles, lighting, and cameras.
    # ------------------------------------------------------------------
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomBrightness(0.2),
    ])

    # ------------------------------------------------------------------
    # 3. Build model: MobileNetV2 backbone + custom head
    # ------------------------------------------------------------------
    # preprocess_input scales pixels from [0,255] to [-1,1] as
    # MobileNetV2 expects (matches its ImageNet pretraining).
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # freeze for phase 1

    inputs  = tf.keras.Input(shape=(224, 224, 3))
    x       = augment(inputs)
    x       = preprocess(x)
    x       = base(x, training=False)
    x       = tf.keras.layers.GlobalAveragePooling2D()(x)
    x       = tf.keras.layers.Dropout(0.3)(x)  # reduce overfitting
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model   = tf.keras.Model(inputs, outputs)

    # ------------------------------------------------------------------
    # 4. Phase 1: train head only
    # ------------------------------------------------------------------
    print(f"\n{'='*50}")
    print("Phase 1: Training classification head (backbone frozen)")
    print(f"{'='*50}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=epochs_head)

    # ------------------------------------------------------------------
    # 5. Phase 2: unfreeze top 30 layers of backbone and fine-tune
    #    Using a much lower LR to avoid destroying pretrained weights.
    # ------------------------------------------------------------------
    print(f"\n{'='*50}")
    print("Phase 2: Fine-tuning top layers of backbone")
    print(f"{'='*50}")

    base.trainable = True
    # Freeze all layers except the last 30
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=epochs_finetune)

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    model.save(model_out)
    print(f"\nModel saved to: {model_out}")

    # Quick val summary
    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f"Final val accuracy: {acc:.3f}  |  val loss: {loss:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames",          default="frames/",
                        help="Directory with fire/ and no_fire/ subfolders.")
    parser.add_argument("--output",          default="models/mobilenetv2_fire.keras",
                        help="Where to save the trained model.")
    parser.add_argument("--epochs-head",     type=int, default=10,
                        help="Epochs for phase 1 (head only).")
    parser.add_argument("--epochs-finetune", type=int, default=10,
                        help="Epochs for phase 2 (fine-tune backbone).")
    args = parser.parse_args()
    train(args.frames, args.output, args.epochs_head, args.epochs_finetune)
