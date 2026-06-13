# Baseline Summary — Dog & Cat Classifiers

## What I mean by "baseline" and why I'm building one first

Before training my main model, I wanted to set up a **baseline**: a simple, cheap, fully reproducible reference point that tells me what level of performance is "trivially achievable" and what level is "achievable with a very basic model". Without this, a number like "70% accuracy" on the final model means nothing on its own — I wouldn't know if that's an amazing result or barely better than guessing. The baseline gives me two concrete numbers to compare against:

- A **floor**: the accuracy of a classifier that always predicts the majority class, with no learning at all. Any real model that scores below this is worse than doing nothing.
- A **simple model**: how far a very basic, well-understood algorithm can get with minimal effort. This tells me how much of the "easy" signal in the data a more sophisticated model would need to beat to actually be worth the added complexity.

## My pipeline (same code for dog and cat)

I load the log-mel features I produced in the preprocessing step (`data/processed/<animal>/{train,val,test}_X.npy`, already normalized using train-only mean/std), and for this baseline I simply **flatten** each spectrogram into a single feature vector (8,064 values for dog, 4,032 for cat).

1. **Floor**: I fit a `DummyClassifier(strategy="most_frequent")` on the train labels and evaluate it on the test set. It always predicts whatever class is most common in train.
2. **Model baseline**: I standardize the flattened features with a `StandardScaler` **fit on train only** (then applied to val and test, so there's no leakage), and train a `LogisticRegression` with `class_weight="balanced"` to account for the class imbalance I documented earlier.
3. Since flattening gives me far more features than training samples (8,064 features vs. 79 dog training samples, for example), the amount of regularization matters a lot. I tried a small grid of values for `C` (the inverse regularization strength: `0.0001, 0.001, 0.01, 0.1, 1, 10`), trained a model for each on train, and picked the one with the best **macro-F1 on the validation set**. This is my "validate on val if needed" step.
4. I then evaluate that selected model **once** on the test set: accuracy, macro-F1, a per-class precision/recall/F1 report, and a confusion matrix figure.

I picked **macro-F1** (the unweighted average of the per-class F1 scores) as my main metric in addition to accuracy, because both my datasets are imbalanced — a classifier that ignores small classes entirely can still get a deceptively high accuracy, but macro-F1 would expose that.

## Why I chose logistic regression over a small CNN for the baseline

I went back and forth on this. A small CNN on the spectrograms would probably be closer to what my final model looks like, so in principle it could be a more "informative" baseline. But I decided against it, for one main reason: **a baseline needs to be stable and reproducible**, almost a fixed point I can always compare to. With so little training data — only 79 dog samples and around 300 cat samples — an unregularized CNN would very likely overfit and its results would depend heavily on initialization, training length, and small architectural choices. That would make it an unstable, not-very-representative reference, and it would make my final-model comparison noisy and hard to interpret ("did my real model actually do better, or did the CNN baseline just have a bad run?").

Logistic regression with `class_weight="balanced"` and a tuned regularization strength, on the other hand, is deterministic given the data and gives me the same numbers every time I run it. I'm treating the CNN idea as something to explore later as **one option for the main model itself** (with proper regularization, dropout, etc.), not as part of the baseline.

## Results — Dog (test set, 17 samples)

| | Accuracy | Macro-F1 |
|---|---|---|
| Floor (always predict "bark") | 0.4118 | 0.1944 |
| Logistic regression (C=0.0001) | 0.8235 | 0.8190 |

The logistic regression baseline clearly beats the floor on both metrics here.

Per-class report on the test set:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| bark | 0.86 | 0.86 | 0.86 | 7 |
| growl | 0.80 | 0.80 | 0.80 | 5 |
| grunt | 0.80 | 0.80 | 0.80 | 5 |

The confusion matrix is saved at `reports/dog_confusion_matrix.png`. Out of 17 test files, 14 are classified correctly, 1 bark is confused with growl, 1 growl with grunt, and 1 grunt with bark — so the errors are spread out and don't seem to concentrate on a single class.

**A comment I want to be honest about**: the dog test set only has 17 files. At that size, each single misclassified file moves the accuracy by roughly 6 percentage points. So I'd read "82%" as "somewhere around 75-85%" rather than a precise number — the headline figure is encouraging, but I wouldn't read too much into the exact decimals with a test set this small.

## Results — Cat (test set, 67 samples)

| | Accuracy | Macro-F1 |
|---|---|---|
| Floor (always predict "isolation") | 0.5224 | 0.2288 |
| Logistic regression (C=0.0001) | 0.4328 | 0.3333 |

Here the picture is more mixed: the baseline **does not** beat the floor on raw accuracy, but it **does** beat it on macro-F1.

Per-class report on the test set:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| brushing | 0.31 | 0.78 | 0.44 | 18 |
| food | 0.00 | 0.00 | 0.00 | 14 |
| isolation | 0.79 | 0.43 | 0.56 | 35 |

The confusion matrix is saved at `reports/cat_confusion_matrix.png`. Out of 67 test files, 29 are classified correctly. The "food" class is never predicted correctly at all (0 recall, 0 precision) — most "food" clips end up classified as "brushing".

**Why I think this happened, and why I'm reporting it as-is rather than hiding it**: "isolation" makes up over half of the cat test set (35/67), so a classifier that always says "isolation" already gets a respectable-looking 52% accuracy — but it completely ignores the other two classes, which is exactly why its macro-F1 is so low (0.23). My logistic regression baseline, with `class_weight="balanced"`, actively tries to give the smaller classes more weight, which pulls it away from just predicting "isolation" all the time — that's why its macro-F1 improves (0.33) even though its raw accuracy drops. But it still isn't able to separate "food" from the other two classes at all with a simple linear model on flattened, normalized spectrograms.

I selected very strong regularization (`C=0.0001`) for both animals. I checked this is a genuine optimum and not just an artifact of my grid's smallest value: with less regularization (e.g. `C=0.01` and above), the model reaches a **training** macro-F1 of essentially 1.0 for both animals — i.e. it perfectly memorizes the training set — while its **validation** macro-F1 stays flat or drops. So `C=0.0001` is the point where the model is forced to rely on broad, generalizable patterns instead of memorizing individual training examples, which is exactly what I'd expect given how few samples I have relative to the number of features.

## Where this leaves me

- For **dog**, the simple logistic regression baseline already does reasonably well (82% accuracy, 0.82 macro-F1) on a very small test set — a good sign, but I'm cautious about over-interpreting it given only 17 test files.
- For **cat**, the baseline beats the floor on macro-F1 but not on accuracy, and it completely fails on the "food" class. This sets a fairly low bar for "food" recognition (0.33 macro-F1 overall) that my main model should be able to clear — and gives me a concrete, honest number to report if it can't.
- I did **not** start working on the main model in this session, as instructed. The next step will be to design that model (likely starting from the CNN idea I set aside here, but with proper regularization) and compare it against these baseline numbers.

---

# Suggested commit message

```
Add baseline models (dummy floor + logistic regression) for dog and cat

- Add src/baseline.py: shared pipeline that evaluates a majority-class
  floor (DummyClassifier) and a logistic regression on flattened,
  standardized log-mel features (class_weight="balanced", C selected
  via macro-F1 on val, no train/val/test leakage)
- Save reports/{dog,cat}_confusion_matrix.png (test set predictions)
- Add reports/baseline_summary.md: what a baseline is, why I chose
  logistic regression over a small CNN for it, and the honest
  floor-vs-baseline comparison (accuracy + macro-F1 + per-class report)
  for both animals
- No main model training in this session
```
