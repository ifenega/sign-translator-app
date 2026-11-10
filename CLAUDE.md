# Sign Translator

Translates sign language from the phone camera. One repo, three parts:

| Folder | What it holds | Stack |
|---|---|---|
| `ml/` | Data and model training | Python 3.13, PyTorch, MediaPipe, OpenCV |
| `server/` | The API | not chosen yet |
| `app/` | Mobile app | React Native |

Goal for all code: clean, readable, easy to understand. Prefer the plain, obvious solution over a clever one.

## ml/

- Virtual environment: `ml/.venv`. Always run Python through it: `ml/.venv/bin/python`, `ml/.venv/bin/pip`.
- `notebooks/` is for exploring only. Anything that will be rerun (data extraction, training, evaluation) goes in a plain `.py` file in `scripts/`. When a notebook experiment works, move the code into a script.
- `data/` holds datasets, extracted landmarks, checkpoints and MediaPipe `.task` model files. It is ignored by git; never commit data or model files.
- Jupyter kernel: **Python (sign-translator)**.
- Train on the Mac GPU with `device="mps"`, and fall back to `"cpu"` when it isn't available.

### Dataset

- Google's Isolated Sign Language Recognition data (Kaggle competition `asl-signs`): about 94k clips of 250 signs from 21 participants, already converted to MediaPipe landmarks. Download with `.venv/bin/kaggle competitions download -c asl-signs -p data/asl-signs`; the login is in `~/.kaggle/access_token`.
- **Never unzip `data/asl-signs/asl-signs.zip`.** It is 37 GB and the unzipped files don't fit on disk alongside it. Scripts read clips straight from the zip.
- The data pipeline is two scripts, run in order from `ml/`:
  1. `scripts/extract_landmarks.py` reads the zip, keeps 105 of the 543 landmarks per frame (lips, both hands, upper body, and the body model's face points) and writes `data/processed/landmarks.npy` and `clips.csv`.
  2. `scripts/preprocess.py` keeps 85 of those points, normalises, mirrors left-handed clips, fixes every clip to 64 frames and splits by participant (the val and test people are listed in the script). It writes `data/prepared/{train,val,test}_{x,mask,y}.npy`. Training reads these.
- Split **by participant**, never at random, so val and test scores show how the model does on signers it has never seen.

### Training

- `scripts/training.py` holds what every model shares: data loading, augmentation, top-1/top-5 accuracy and the training loop. Each model gets its own `scripts/train_<model>.py` that defines the model and calls `train()`.
- Every model takes the same input, `(x, mask)`: points of shape `(clips, 64, 85, 2)` and a mask that is True where a point is missing.
- Compare models on **val**. Leave **test** untouched until the final model is chosen.
- Checkpoints and per-epoch history go to `data/checkpoints/<name>.pt` and `<name>_history.csv`.
- `scripts/evaluate.py <name> [<name> ...]` breaks a Transformer checkpoint's val score down: all clips, without the cut-off clips of 6 frames or fewer (about 12% of the data), and per person. Given several checkpoints, it averages their probabilities and scores the combination.

### Dependencies

- `requirements.txt` lists the top-level packages; `requirements.lock.txt` pins exact versions. After installing or removing anything, run `.venv/bin/pip freeze > requirements.lock.txt`.
- **OpenCV:** only `opencv-contrib-python` is installed (MediaPipe depends on it). Never add `opencv-python` as well: both packages install to the same `cv2` folder, and uninstalling either one breaks the other.
- **MediaPipe 1.x:** the old `mp.solutions` API (`mp.solutions.holistic`, `mp.solutions.hands`) **no longer exists**. Use the Tasks API in `mediapipe.tasks.python.vision`: `HandLandmarker`, `PoseLandmarker`, `FaceLandmarker`. Each one needs its `.task` model file in `ml/data/`.

## app/ — feature-based structure

Expo SDK 58 with Expo Router and TypeScript. Run commands from `app/`: `npx expo start` to run it, `npx expo lint` and `npx tsc --noEmit` before calling anything done, and `npx expo install <package>` (never plain `npm install`) so versions match the SDK. `app/AGENTS.md` has Expo's own notes.

The app is organised by **feature** (also called feature-first or "vertical slice" structure): everything for one feature lives together in one folder, rather than all hooks in one place and all API calls in another.

```
app/src/
├── app/                  # Expo Router routes only, kept thin: each route renders a screen from a feature
├── features/
│   └── <feature>/        # e.g. translate, camera, history, auth
│       ├── api/          # calls to the server for this feature
│       ├── components/   # UI used only by this feature
│       ├── hooks/
│       ├── screens/
│       ├── utils/
│       ├── types.ts
│       └── index.ts      # the feature's public API
└── shared/               # anything used by 2 or more features
    ├── api/              # HTTP client, base config
    ├── components/       # buttons, inputs, layout pieces
    ├── hooks/
    ├── utils/
    ├── theme/            # colours, spacing, typography
    └── types/
```

Rules:

1. **Code starts inside its feature.** New code goes in the feature that uses it.
2. **Move to `shared/` once a second feature needs it.** When a second feature needs something, move it to `shared/` instead of copying it or importing it from the other feature.
3. **Features don't reach into each other.** A feature imports from another feature only through that feature's `index.ts`, never from its internal files. If two features need the same thing, it belongs in `shared/`.
4. **`shared/` never imports from `features/`.** Dependencies point one way: `app/` → `features/` → `shared/`.
5. **Only create the subfolders a feature needs.** A feature with no API calls has no `api/` folder; don't add empty folders.
6. **Routes stay thin.** Files in `src/app/` hold no logic. They import and render a screen from a feature.
7. **Use readable names.** Feature folders are lowercase and named after what the user does (`translate`, not `module1`). Hooks start with `use`. Each component gets its own file, named in PascalCase.

## server/

Not started. When it is, add its structure and run commands here.
