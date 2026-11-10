# Sign Translator

Translates sign language from the phone. The phone records the signer, the hands, lips, face and upper body are tracked with MediaPipe, and a model turns those movements into words.

| Folder | What it holds | Stack |
|---|---|---|
| `ml/` | Data and model training | Python 3.13, PyTorch, MediaPipe, OpenCV |
| `server/` | The API | FastAPI (planned) |
| `app/` | Mobile app | React Native, Expo SDK 58, Expo Router |

## Where the project is

The sign recognition model's settings are chosen and **the final models are training**. The app's first version is built, but it shows a placeholder answer until the server exists.

| Step | Status |
|---|---|
| Get the dataset | Done |
| Extract and prepare landmarks | Done |
| Train a baseline, then improve it | Done |
| Train the final models and score them on the test set | **In progress** |
| Server: `POST /predict` takes a video and returns the top signs | Not started |
| Mobile app | First version built, waiting for the server |

### Model results

Scores on signers the model has never seen (val split, 250 signs). Top-1 means the right sign was the model's first guess; top-5 means it was in its top five.

| Model | Top-1 | Top-5 |
|---|---|---|
| Convolution + Transformer, with motion input, augmentation and label smoothing | 72.2% | 91.1% |
| + extra augmentation that imitates different people (hand size and angle, hidden lips) | 72.3% | 91.4% |
| + 11 face points (nose, eyes, ears, mouth corners) | 72.7% | 91.3% |
| A bigger version of the same model | 71.7% | 91.1% |
| The face-points model trained for 100 epochs instead of 40 | 73.8% | 91.3% |
| **3 copies of the face-points model (40 epochs each) combined** | **74.0%** | |

Combining three copies of the model and averaging their answers helps more than making the model bigger, so the final model is three copies trained for 100 epochs each. Without the cut-off clips of 6 frames or fewer (about 12% of the data), the best setups reach 77.0% top-1.

The target is **80% top-1 or better**. The weakest case is the left-handed signer in val, at about 66%. The models learn the training signers much better than new ones, so more variety of people helps more than a bigger model.

Earlier models were scored on a first val split that held only left-handed signers, so their scores can't be compared with the table above: a mean/std baseline with an MLP reached 54.6% top-1, a bidirectional GRU 60.3%, and the first Transformer 64.0%.

### Final models

`final_seed0`, `final_seed1` and `final_seed2` are trained on train + val together (18 people) for 100 epochs each, about 1.5 hours per model. When all three are done, they are scored on the test split once. Nothing is tuned against test after that.

### App

The first version is in `app/`:

- **Record** a 3-second video with the front camera (an outline shows where to stand, then a 3-2-1 countdown), or **choose** a video from the phone.
- The video plays in a card while it's translated. One word shows on a lime card; a sentence shows on a plain card.
- Until the server exists, every video comes back as "Hello" (86% confident). The result and confidence are logged in the terminal.

## Dataset

Google's [Isolated Sign Language Recognition](https://www.kaggle.com/competitions/asl-signs) data from Kaggle: about 94,000 clips of 250 American Sign Language signs from 21 people, already converted to MediaPipe landmarks.

The data is split **by person**, not at random: validation and test clips come from people who aren't in the training set, so the scores show how the model does on new signers. Val and test each hold 2 right-handed and 1 left-handed signer.

## Setup (ml/)

```sh
cd ml
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock.txt   # exact versions; requirements.txt lists the top-level packages
```

Download the data (needs a Kaggle account and the competition rules accepted):

```sh
.venv/bin/kaggle competitions download -c asl-signs -p data/asl-signs
```

The zip is 37 GB. **Don't unzip it**; the scripts read straight from it.

## Running the pipeline

All commands run from `ml/`.

```sh
# 1. Pull the 105 landmarks we use out of the zip -> data/processed/
.venv/bin/python scripts/extract_landmarks.py

# 2. Keep 85 points, normalise, mirror left-handed clips, fix every clip to 64 frames, split by person -> data/prepared/
.venv/bin/python scripts/preprocess.py

# 3. Train a model -> data/checkpoints/<name>.pt and <name>_history.csv
.venv/bin/python scripts/train_transformer.py --name transformer_face --epochs 40 --label-smoothing 0.1

# 4. Break its val score down: overall, without cut-off clips, and per person.
#    Several names are combined by averaging their answers.
.venv/bin/python scripts/evaluate.py transformer_face
```

Training uses the Mac GPU (`mps`) when it's available and the CPU otherwise. Close the Android emulator while training: on a 16 GB Mac, the two together run out of memory and training slows to a crawl.

## ml/ layout

- `scripts/`: everything that gets rerun. `training.py` holds what all models share (data loading, augmentation, accuracy, the training loop); each `train_<model>.py` defines one model; `evaluate.py` scores checkpoints.
- `notebooks/`: exploration only. Use the **Python (sign-translator)** kernel; start Jupyter with `.venv/bin/jupyter lab`.
- `data/`: datasets, landmarks and checkpoints. Ignored by git.

## Running the app (app/)

```sh
cd app
npm install
npx expo start
```

Scan the QR code with the **Expo Go** app on a phone on the same Wi-Fi, or press `a` for an Android emulator. Before calling a change done, run `npx expo lint` and `npx tsc --noEmit`. Add packages with `npx expo install <package>` so their versions match the Expo SDK.

The app is organised by feature: `src/features/translate` (choosing a video and showing the result) and `src/features/record` (the camera). `src/app/` only holds thin routes, and `src/shared/` holds what several features use, such as the theme.
