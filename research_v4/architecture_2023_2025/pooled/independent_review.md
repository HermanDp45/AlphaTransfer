# Независимая проверка pooled TabM / HGB

Статус **PASS**. Проверены все временные границы 2023–2025 и 6 завершённых пар checkpoints (по одному TabM/HGB на feature-set/year). Экспериментальный код и исходные данные не изменены.

Числовые входы и индикаторы missing совпадают побитно в float32. Median imputer и quantile transformer независимо переобучены только на outer train: statistics, quantiles и references совпадают. HGB получает тот же числовой блок плюс one-hot пяти коридоров; TabM — те же категории через встроенный parameter-free _OneHotEncoding; результат также совпадает побитно. Обучаемые PeriodicEmbeddings относятся только к числовым входам.

Даты созревания проверены независимо как пятое следующее наблюдение внутри каждой валюты. Train/validation/test labels строго раньше соответствующей следующей границы; внутренний split TabM тоже purged. У незрелого history хвоста target и все три future utility поля пусты. Для завершённых моделей заново рассчитаны raw probabilities из checkpoints для validation, history и test; архитектуры имеют одинаковые date/corridor/labels.

## Ограничения интерпретации

- Same external preprocessing and feature information do not mean identical inductive bias: TabM has learned numerical embeddings and an ensemble head (its corridor encoding is exactly the same one-hot), inner epoch selection; HGB has fixed120trees and one-hot corridor. This is a fair fixed-recipe comparison, not matched HPO or general architecture ranking.
- Existing TabM model.json triggers load without verifying current features/seed/training fingerprint. New checkpoint directory and present metadata match this run; if code or inputs change, clear cache or validate full input/config hashes before resume.
- Five corridor observations on the same market date are correlated; pooled-row confidence intervals would overstate independent sample size. Use paired date/month blocks and keep per-corridor calibration/policy identical.
- No2026 outcomes are exported or evaluated, but the fixed recipes/features were inherited from previous retrospective research. Treat 2023–2025 as a retrospective comparison, not a preregistered pristine holdout.

Детальные hashes, сроки maturity, число проверенных строк и погрешности: `independent_review.json`.
