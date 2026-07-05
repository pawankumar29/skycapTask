# Indian Currency Note Detector

A local web app for checking Indian currency note images. It estimates the denomination and returns a binary Accepted or Rejected result.

Supported denominations:

- Rs. 10
- Rs. 20
- Rs. 50
- Rs. 100
- Rs. 200
- Rs. 500
- Rs. 2000

The app runs with Flask, OpenCV, NumPy, and an optional TensorFlow classifier. Docker is the recommended way to build, train, and run the project. No external API key or hosted service is required at runtime.

## Docker First Setup

After cloning the repo, copy the dataset separately into this structure:

```text
currency/data/fake/
currency/data/real/
currency/denomination_data/
```

Build and run the app:

```bash
docker compose up --build
```

Open:

```text
http://localhost:8080
```

Stop the app:

```bash
docker compose down
```

The app can run without a trained TensorFlow model. In that case it uses the OpenCV and reference-image checks only. For better fake-note detection, train the TensorFlow model and then rebuild the app.

## Dataset Guidelines

Model accuracy depends heavily on dataset quality. A better dataset will usually improve real/fake detection more than code changes alone.

You can collect images from Kaggle or any other suitable platform according to the project requirement. You can also add your own camera photos. Make sure every image is legally usable for the project and is placed in the correct folder.

Keep the dataset balanced:

- Keep the number of `real` and `fake` images as close as possible.
- Keep all denominations represented: Rs. 10, Rs. 20, Rs. 50, Rs. 100, Rs. 200, Rs. 500, and Rs. 2000.
- Avoid training with many images from one denomination and very few from another.
- Add front-side, back-side, cropped, rotated, blurred, low-light, bright-light, and normal camera images.
- Keep separate test images that are not copied from the training images.

Recommended real/fake dataset structure for TensorFlow:

```text
currency/data/real/
currency/data/fake/
```

Recommended denomination dataset structure:

```text
currency/denomination_data/10/
currency/denomination_data/20/
currency/denomination_data/50/
currency/denomination_data/100/
currency/denomination_data/200/
currency/denomination_data/500/
currency/denomination_data/2000/
```

The `currency/data/real/` and `currency/data/fake/` folders are used for authenticity training. The `currency/denomination_data/` folders are used when training or improving denomination-specific detection.

## Denomination Dataset

Use this folder when adding training images for the TensorFlow denomination classifier:

```text
currency/denomination_data/10/
currency/denomination_data/20/
currency/denomination_data/50/
currency/denomination_data/100/
currency/denomination_data/200/
currency/denomination_data/500/
currency/denomination_data/2000/
```

Put note images into the folder matching the note value. For example, Rs. 500 images should go into `currency/denomination_data/500/`.

This dataset is also ignored by Git because it can become large. Share it separately with whoever will train or test the project.

## Train TensorFlow In Docker

TensorFlow trains two local classifiers:

- authenticity classifier: `real` vs `fake`
- denomination classifier: Rs. 10, Rs. 20, Rs. 50, Rs. 100, Rs. 200, Rs. 500, Rs. 2000

The datasets are not committed to Git because they are large.

Accuracy depends heavily on the dataset used for training. Better real/fake coverage, balanced denomination folders, and more training epochs will usually improve results. Weak or missing training data for a denomination can reduce accuracy even when the code is working correctly.

Required authenticity dataset structure:

```text
currency/data/fake/
currency/data/real/
```

Required denomination dataset structure:

```text
currency/denomination_data/10/
currency/denomination_data/20/
currency/denomination_data/50/
currency/denomination_data/100/
currency/denomination_data/200/
currency/denomination_data/500/
currency/denomination_data/2000/
```

Then train inside Docker:

```bash
docker compose run --rm trainer
```

This command trains the real/fake model. It also trains the denomination model only when all denomination folders exist and contain images. If `currency/denomination_data/` is missing or incomplete, denomination training is skipped and the real/fake model still trains normally.

To force only the real/fake model, run:

```bash
docker compose run --rm trainer python train_tensorflow_model.py --skip-denomination
```

For a quick low-memory training run:

```bash
docker compose run --rm trainer python train_tensorflow_model.py --epochs 2 --batch-size 4 --prefetch 1
```

Training creates:

- `currency_authenticity_model.keras`
- `currency_authenticity_model.json`
- `currency_denomination_model.keras`
- `currency_denomination_model.json`

These generated model files are also ignored by Git. After training, rebuild the app locally so the model files are copied into the image:

```bash
docker compose up --build
```

Full recommended Docker flow:

```bash
docker compose up --build
docker compose down
docker compose run --rm trainer
docker compose up --build
```

## Optional: Run Without Docker

Docker is the recommended way to run this project. Use this section only if you want to run the Flask server directly on your machine for local debugging.

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the server:

```bash
.\.venv\Scripts\python.exe server.py
```

Open:

```text
http://localhost:8080
```

## What The App Checks

The backend combines three local checks:

1. Denomination and reference matching
   - Crops the note foreground.
   - Converts the image into compact color and grayscale vectors.
   - Compares the uploaded image with local real/fake reference images.
   - Detects whether the closest neighbors are mostly genuine or fake.

2. OpenCV feature checks
   - Uses ORB feature matching.
   - Uses SSIM-style similarity checks.
   - Checks edge/bleed-line signals.
   - Checks serial/print texture.
   - Runs a generated 10-check pipeline for Rs. 10, Rs. 20, Rs. 50, Rs. 100, and Rs. 200.
   - Keeps the original notebook-style pipeline for Rs. 500 and Rs. 2000.

3. TensorFlow fake/real classifier
   - Loads `currency_authenticity_model.keras` when available.
   - Runs a CNN-based softmax classifier.
   - Returns fake/real probabilities.
   - Gives TensorFlow high priority when fake probability is strong.
   - The final decision is binary: accepted notes pass the combined checks, all other notes are rejected.
   - If no trained model is present, the app still works using OpenCV/reference checks.

4. TensorFlow denomination classifier
   - Loads `currency_denomination_model.keras` when available.
   - Uses `currency/denomination_data/` during training.
   - Predicts Rs. 10, Rs. 20, Rs. 50, Rs. 100, Rs. 200, Rs. 500, or Rs. 2000.
   - Supports the reference/OpenCV denomination result when confidence is high.

## TensorFlow Model

The TensorFlow models are CNN image classifiers, not regression models.

Authenticity output classes:

- `fake`
- `real`

Denomination output classes:

- `10`
- `20`
- `50`
- `100`
- `200`
- `500`
- `2000`

Final layer:

```python
tf.keras.layers.Dense(2, activation="softmax")
```

Loss:

```python
sparse_categorical_crossentropy
```

Recommended Docker training command:

```bash
docker compose run --rm trainer
```

Optional local training command, only if TensorFlow is installed on your machine:

```bash
python train_tensorflow_model.py
```

## Source Files

- `index.html` - app layout
- `styles.css` - UI styling
- `app.js` - upload, preview, and result rendering
- `server.py` - Flask API server
- `detector.py` - OpenCV/reference/TensorFlow detection logic
- `train_tensorflow_model.py` - TensorFlow trainer
- `requirements.txt` - Python dependencies
- `Dockerfile` - app image definition
- `docker-compose.yml` - app and trainer services
- `reference_cache.npz` - compact local reference vectors
- `Features/Features/` - feature templates
- `Project_files/` - legacy notebook/reference material

Not included in Git:

- `currency/data/` - large real/fake training dataset
- `currency/denomination_data/` - large denomination training dataset
- `currency_authenticity_model.keras` - generated TensorFlow model
- `currency_authenticity_model.json` - generated model metadata
- `currency_denomination_model.keras` - generated denomination TensorFlow model
- `currency_denomination_model.json` - generated denomination model metadata

## Git Notes

Unneeded generated files are ignored with `.gitignore`, including:

- `__pycache__/`
- `.venv/`
- `node_modules/`
- logs and temp files
- IDE folders
- `currency/`
- `currency_authenticity_model.keras`
- `currency_authenticity_model.json`
- `currency_denomination_model.keras`
- `currency_denomination_model.json`

## Limitations

This is a lightweight local detector, not a certified banknote authentication system. Results depend on:

- image quality
- lighting
- crop and rotation
- front-side vs back-side note image
- fake/real dataset quality
- how well the TensorFlow model is trained
- number of TensorFlow training epochs

## What Can Be Done Better

- Train the TensorFlow model for 20-25 epochs when time and memory allow.
- Add more real and fake samples for every denomination.
- Keep the fake and real datasets balanced.
- Add rotated, cropped, blurred, low-light, front-side, and back-side examples.
- Improve automatic note rotation before detection.
- Build a cleaner validation report for each denomination.
- Tune thresholds separately for Rs. 10, Rs. 20, Rs. 50, Rs. 100, Rs. 200, Rs. 500, and Rs. 2000.
- Improve handling for highly cropped, low-quality, or multi-note images.

## Commit And Push

```bash
git status
git add .gitignore README.md app.js detector.py docker-compose.yml Dockerfile index.html requirements.txt server.py styles.css train_tensorflow_model.py reference_cache.npz Features Project_files
git commit -m "Improve fake currency detector"
git push
```
