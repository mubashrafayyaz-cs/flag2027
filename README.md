# FAST-Fair: face–voice association for FLAG 2027

Code for team **FAST-Fair** (CodaBench user `mubashrafayyaz`) in the FLAG 2027 Grand Challenge (Face–voice Association across LAnguages and Gender, IEEE ICASSP 2027).

All encoders are public and frozen. We train only a small two-branch head (FOP-style) on top of a CLIP ViT-L/14 face embedding and a stack of six frozen voice embeddings, with a language-nuisance projection and a supervised contrastive alignment loss. Details are in `docs/system_description.pdf`.

## Results

| Submission | no-gender En | no-gender Bn | gender En | gender Bn | mean EER |
|---|---|---|---|---|---|
| **Evaluation set, avg(E, E2), ranked submission** | 18.88 | 16.21 | 25.96 | 32.68 | **23.43** |
| Evaluation set, E2 alone | 18.88 | 16.21 | 26.44 | 32.68 | 23.55 |
| Progress set, E2 | 16.07 | 15.36 | 29.12 | 27.11 | 21.92 |
| Progress set, E | 16.47 | 16.22 | 29.53 | 26.43 | 22.16 |
| Organizer FOP baseline (progress) | 32.54 | 38.12 | 32.99 | 44.01 | 36.92 |

The ranked submission is the plain, equal-weight average of the scores of systems E and E2. Both systems and the averaging rule were fixed before the evaluation data was released.

## Repository layout

```
notebooks/
  20_final_pipeline.ipynb   final pipeline: trains the heads (if missing) and scores raw files -> submission zip
  01-19, 21-46              feature extraction and every experiment reported in the system description
checkpoints/                trained heads (download from the GitHub Release, see below)
docs/system_description.pdf
requirements.txt
```

## Environment

Everything ran on **Google Colab** with one GPU (T4 or L4) and Google Drive for storage. The notebooks expect this Drive layout:

```
MyDrive/FLAG2027/
  paths.json        {"ZIP_DIR": "<Drive folder holding the official FLAG 2027 train and dev zips>"}
  features/         cached training features (written by the extraction notebooks)
  final/            trained heads head_<system>_s21.pt
  eval/             exactly ONE evaluation zip (e.g. evaluation_phase.zip)
  submissions/      output zips
  results/          scores and cross-validation logs
  models_cache/     downloaded model weights
```

Packages: Colab's default PyTorch/transformers stack plus `speechbrain onnx onnx2torch peft` (installed by Cell 1 of the notebook). `requirements.txt` lists the exact versions frozen from the Colab runtime used for the evaluation run. A Hugging Face token (Colab secret `HF_TOKEN`) is needed only for gated models in some experiment notebooks, not for the final pipeline.

## Reproduce the ranked submission from the trained heads

1. Download `head_E_s21.pt` and `head_E2_s21.pt` from this repository's Release and put them in `MyDrive/FLAG2027/final/`.
2. Put the organizers' `evaluation_phase.zip` (alone) in `MyDrive/FLAG2027/eval/`.
3. Open `notebooks/20_final_pipeline.ipynb` in Colab with a GPU. Cell 1 is set to `MODE = 'eval'`, `EVAL_ENTRY = 'E2'`. Run all cells up to Cell 8. This extracts all features from the raw evaluation files and writes `submissions/20_eval_evaluation_phase_s21_E2.zip`.
4. Set `EVAL_ENTRY = 'E'` in Cell 1 and run all again (features are cached, a few minutes).
5. Run Cell 9. It averages the saved E and E2 scores and writes `submissions/20_eval_evaluation_phase_s21_E_E2.zip`, the ranked submission.

Each trial is scored independently: the score is the L2 distance between the face and voice embeddings (lower = same person). There is no score normalisation, calibration, test-time adaptation or cross-trial processing. A dry run on the progress set reproduced our uploaded scores at correlation 1.0000 in all four cells.

Note on the evaluation lists: in `evaluation_phase/gender/English_valid.txt` every face path has a space instead of a slash (`English_valid/faces 00000.jpg`). Cell 5 rejoins such paths and accepts a repaired path only if that file exists.

## Reproduce training

The heads are trained by Cell 3 of `20_final_pipeline.ipynb` from cached features; it trains any head that is missing from `final/` (21 seeds per system, about 12 s per head on a T4). The cached features are produced by these notebooks, in order:

| Notebook | Produces |
|---|---|
| 01–05 | checks of the official data; challenge-row face (CLIP) and voice features; the official ECAPA CSVs are read directly |
| 06 | MAV-Celeb v1 and v3 rows (organizer-approved external data) |
| 07 | VoxCeleb2 test zip + dev part 1 rows, one face frame per clip, Bengali clips removed with VoxLingua107 |
| 12, 13 | WeSpeaker ResNet293 features for all sources |
| 14 | MAV-Celeb v2 rows |
| 17, 18 | ReDimNet2 and audeering wav2vec2 age–gender features for all sources |
| 26 | Whisper-large-v3 encoder features, MAV-Celeb language labels for the language-nuisance projection |
| 32 | VoxCeleb2 dev part 2 rows with all encoders (used by system E2), Bengali removed |

System definitions are in Cell 2 of the final notebook: `E` = six voice blocks + CLIP, NAP with 8 directions, supervised contrastive (L_out) alignment with weight 0.5 and temperature 0.07, 64 identities × 8 clips per batch, cosine schedule; `E2` = `E` plus VoxCeleb2 dev part 2.

## Compliance summary

- Progress and evaluation data were used for inference only: no training, adaptation, pseudo-labels, statistics or calibration.
- No Bengali speech was used to train or adapt any audio encoder. All audio encoders are frozen public models; Bengali clips were removed from VoxCeleb2 with VoxLingua107 before training our head.
- External data: MAV-Celeb v1–v3 (approved by the organizers) and VoxCeleb2, both public.
- Model selection used cross-validation on the official training data; the progress leaderboard was used only to choose among finished systems.

## Experiments

Notebooks 21–46 are the experiments behind Section 6 of the system description (what did not help), including encoder fine-tuning (40, 40b, 41), other face encoders (23, 35, 45), other voice features (21, 24, 29, 39, 44), public cross-modal models (42), loss and training variants (27, 28, 31, 33, 34, 36, 37, 38, 46) and data variants (32b, 32c). Each "runall" notebook prints its own result summary.

## Authors

Mubashra Fayyaz (FAST-NUCES Karachi), Rauf Ahmed Shams Malick (Al-Ghazali University). Contact: mubashrafayaz.cs@gmail.com

AI-assisted tools were used for code development and editing.

## License

MIT (see `LICENSE`). Pretrained models and datasets keep their own licences.
