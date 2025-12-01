# Tello Gesture ML

Prototype pipeline for classifying a small set of drone gestures from webcam frames. It currently relies on the [Sign Language Digits Dataset](https://github.com/ardamavi/Sign-Language-Digits-Dataset) to train a TensorFlow image classifier that maps seven static hand poses to drone commands (`takeoff`, `land`, `forward`, `backward`, `left`, `right`, `spin`). The laptop webcam can then be used for quick experiments by feeding frames through the trained network.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset preparation

The helper script downloads and reshapes the dataset into `data/processed/{train,val}` while storing metadata in `data/processed/label_mapping.json`.

```bash
python src/data/prepare_dataset.py --raw-dir data/raw --processed-dir data/processed
```

Only digits `0-6` are used and re-labeled into drone commands:

| Digit | Gesture  |
| ----- | -------- |
| 0     | takeoff  |
| 1     | land     |
| 2     | forward  |
| 3     | backward |
| 4     | left     |
| 5     | right    |
| 6     | spin     |

## Training

```bash
python src/train.py --data-dir data/processed --epochs 15 --batch-size 64
```

The script saves the best checkpoint to `models/gesture_cnn.keras` alongside `models/class_to_idx.json` for inference. TensorBoard logs are not included yet but the script prints per-epoch metrics.
After each training run, `training_curves.png` is written to the specified `--output-dir`, showing train vs. validation accuracy/loss so you can catch overfitting.

### Model variants and pretrained checkpoints

Model builders live under `src/build_models`. Use `--model-version` to pick an internal architecture:

```bash
python src/train.py --model-version model_v2
```

`model_v3` swaps the CNN for a PCA + Logistic Regression pipeline; it fits PCA/logistic once, exports a frozen `.keras` model, and prints one-epoch metrics/plots:

```bash
python src/train.py --model-version model_v3
```

To start from a local `.keras` checkpoint, point `--pretrained-path` at the file:

```bash
python src/train.py --pretrained-path models/mobilevit_finetune.keras
```

You can also download `.keras` weights directly from Hugging Face:

```bash
python src/train.py \
  --hf-repo keras-io/mobilevit-xxs \
  --hf-filename mobilevit_xxs_classifier.keras
```

An optional `--hf-cache-dir` overrides the default Hugging Face cache location.

Need to troubleshoot labels? Add `--debug-data` to print the inferred class order plus sample batches:

```bash
python src/train.py --debug-data --output-dir models
```

Sample grids are exported to `models/debug_samples/debug_samples_{train,val}.png`. Adjust `--debug-num-images` for more/less samples.

## Evaluation

Measure accuracy, precision, recall, F1, and generate a confusion matrix plot on the validation split:

```bash
python src/test.py --data-dir data/processed --split val --model-path models/gesture_cnn.keras
```

By default the confusion matrix figure is stored in `models/confusion_matrix_val.png`.

## Quick webcam loop

After training, you can run a simple webcam loop that crops the center of the frame and feeds it through the trained model:

```bash
python src/live_demo.py --model-path models/gesture_cnn.keras --label-path models/class_to_idx.json
```

This is intentionally barebones—you will still need to implement drone control logic (e.g., via `djitellopy`) and a more robust ROI extraction strategy for reliable control.

## Next steps

- Collect domain-specific gesture data with the actual webcam and drone operator
- Experiment with MediaPipe-based hand landmark extraction for better generalization
- Replace the static classifier with a temporal model when dynamic gestures are needed
- Integrate directly with the DJI Tello SDK after validating predictions offline

## Dataset license

The Sign Language Digits Dataset is distributed under the MIT License by its authors; review that license before redistributing.
