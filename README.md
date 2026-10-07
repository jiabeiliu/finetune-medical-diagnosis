# Fine-Tuning gpt-4o-mini for Medical Diagnosis — and What Went Wrong

Coursework project (INFO7374, Northeastern University, Spring 2026 — scored 20/20).
An end-to-end fine-tuning experiment using OpenAI's managed fine-tuning platform,
with a focus on **honest evaluation**: the fine-tuned model got *worse*, and
diagnosing why was the most valuable part.

## What I did

1. **Data prep** — Took the provided fine-tuning / validation / test sets (clinical
   case vignettes with gold diagnoses), prepended task instructions to every query,
   and converted CSV → JSONL for the fine-tuning API.
2. **Fine-tuning** — Fine-tuned `gpt-4o-mini` via `platform.openai.com/finetune`
   under **two TA-assigned hyperparameter configurations**; tracked and compared
   the training-loss curves for both runs.
3. **Evaluation harness** — Built a 20-case test set split into *New Samples*
   (unseen cases) and *Finetuned Samples*; for each case compared the model's
   first diagnosis against the gold diagnosis before and after fine-tuning.

## Results

| | Correct / 20 | Accuracy |
|---|---|---|
| Before fine-tuning | 11 | 55% |
| After fine-tuning | 1 | 5% |

The fine-tuned model **collapsed to 3 repeated diagnoses** across all 20 cases
(6× neurosyphilis, 6× colonic paragonimiasis, 8× septic arthritis variants) —
a textbook case of **mode collapse / overfitting** to the fine-tuning
distribution. Reproduce it:

```bash
python eval/evaluate.py
```

## Takeaways

- Fine-tuning on a narrow distribution can destroy a generalist model's
  calibration — always keep a held-out eval with *new* samples, not just
  fine-tuned ones.
- Training-loss curves alone don't tell you the model is still useful;
  behavioral eval before/after is the real gate.
- A "failed" experiment with a clean diagnosis is worth more than a lucky
  accuracy bump you can't explain.

## Data note

Case vignettes are de-identified teaching cases provided with the course;
`results/results.csv` contains my experimental outputs (model responses and
Correct/Wrong verdicts) without the full vignette text. No PII, no API keys.
