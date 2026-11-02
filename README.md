# Sign Translator

Translates sign language from the phone camera. The plan: the phone tracks the signer's hands, lips and upper body with MediaPipe, and a model turns those movements into words.

| Folder | What it holds | Stack |
|---|---|---|
| `ml/` | Data and model training | Python 3.13, PyTorch, MediaPipe, OpenCV |
| `server/` | The API | not chosen yet |
| `app/` | Mobile app | React Native |

## Where the project is

Early stage: **training the sign recognition model.** The app and server have not been started.

| Step | Status |
|---|---|
| Get the dataset | Done |
| Extract and prepare landmarks | Done |
| Train a baseline, then improve it | **In progress** |
| Pick the final model and score it on the test set | Not started |
| Server | Not started |
| Mobile app | Not started |

Best results so far, on signers the model has never seen (val split, 250 signs):

| Model | Script | Top-1 | Top-5 |
|---|---|---|---|
| Baseline: mean/std of each point + MLP | `train_baseline.py` | 54.6% | 79.1% |
| Bidirectional GRU | `train_gru.py` | 60.3% | 81.5% |
| Convolution + Transformer, with motion input | `train_transformer.py` | 64.0% | 84.1% |

A longer Transformer run (116 epochs, extra augmentation, label smoothing) reached 65.0% top-1 but levelled off after about epoch 60.

These scores were measured on the first val split, which happened to hold 3 left-handed signers (the hardest case). Val has since been redrawn to 2 right-handed and 1 left-handed signer, like test, so the models need retraining before their scores can be compared on it.

Top-1 means the right sign was the model's first guess; top-5 means it was in its top five. The target is **80% top-1 or better**. The models learn the training signers well (96% on train) but do much worse on new signers, so the current work is about generalising to people the model hasn't seen.

## Dataset

Google's [Isolated Sign Language Recognition](https://www.kaggle.com/competitions/asl-signs) data from Kaggle: about 94,000 clips of 250 American Sign Language signs from 21 people, already converted to MediaPipe landmarks.

The data is split **by person**, not at random: validation and test clips come from people who aren't in the training set, so the scores show how the model does on new signers.

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
# 1. Pull the 94 landmarks we use out of the zip -> data/processed/
.venv/bin/python scripts/extract_landmarks.py

# 2. Normalise, mirror left-handed clips, fix every clip to 64 frames, split by person -> data/prepared/
.venv/bin/python scripts/preprocess.py

# 3. Train a model -> data/checkpoints/<name>.pt and <name>_history.csv
.venv/bin/python scripts/train_transformer.py --name transformer_motion --epochs 40
```

Training uses the Mac GPU (`mps`) when it's available and the CPU otherwise.

## ml/ layout

- `scripts/`: everything that gets rerun. `training.py` holds what all models share (data loading, augmentation, accuracy, the training loop); each `train_<model>.py` defines one model.
- `notebooks/`: exploration only. Use the **Python (sign-translator)** kernel; start Jupyter with `.venv/bin/jupyter lab`.
- `data/`: datasets, landmarks and checkpoints. Ignored by git.
