# Indian Currency Note Detector

A local web app for checking Indian currency note images. It estimates the denomination and rejects notes that look fake or fail authenticity checks.

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

## Train TensorFlow In Docker

The TensorFlow model is used as a strong fake-note signal. The dataset is not committed to Git because it is large.

Required dataset structure:

```text
currency/data/fake/
currency/data/real/
```

Then train inside Docker:

```bash
docker compose run --rm trainer
```

For a quick low-memory training run:

```bash
docker compose run --rm trainer python train_tensorflow_model.py --epochs 2 --batch-size 4 --prefetch 1
```

Training creates:

- `currency_authenticity_model.keras`
- `currency_authenticity_model.json`

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
   - If no trained model is present, the app still works using OpenCV/reference checks.

## TensorFlow Model

The TensorFlow model is a CNN image classifier, not a regression model.

Output classes:

- `fake`
- `real`

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
- `currency_authenticity_model.keras` - generated TensorFlow model
- `currency_authenticity_model.json` - generated model metadata

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

## Limitations

This is a lightweight local detector, not a certified banknote authentication system. Results depend on:

- image quality
- lighting
- crop and rotation
- front-side vs back-side note image
- fake/real dataset quality
- number of TensorFlow training epochs

## What Can Be Done Better

- Train the TensorFlow model for 20-25 epochs when time and memory allow.
- Add more real and fake samples for every denomination.
- Keep the fake and real datasets balanced.
- Add rotated, cropped, blurred, low-light, front-side, and back-side examples.
- Improve automatic note rotation before detection.
- Build a cleaner validation report for each denomination.
- Tune thresholds separately for Rs. 10, Rs. 20, Rs. 50, Rs. 100, Rs. 200, Rs. 500, and Rs. 2000.
- Add a manual review state for highly cropped or low-quality images.

## Commit And Push

```bash
git status
git add .gitignore README.md app.js detector.py docker-compose.yml Dockerfile index.html requirements.txt server.py styles.css train_tensorflow_model.py reference_cache.npz Features Project_files
git commit -m "Improve fake currency detector"
git push
```
