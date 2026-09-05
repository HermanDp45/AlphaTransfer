# Ограниченный age-aware follow-up

Эта гипотеза добавлена после просмотра основного continuation, включая ранее известный 2026. Она не заменяет первоначальную selection и не объявляется preregistered holdout. Предобъявлены ровно две политики:

- `age_fallback_v3`: сохранить Halyk lag1 probability, когда наблюдаемые Halyk данные достаточно свежие; иначе использовать V3 global-calibrated probability.
- `age_fallback_minimax`: тот же gate, fallback в уже выбранную minimax blend policy.

Age — максимальный calendar age доступных observation dates для personal RUB, legal RUB и personal USD, используемых Halyk features. Отсутствие любого необходимого observation date считается unavailable. Gate не получает normal/delayed flag. Cutoff age выбирается из {2,3} calendar days на предыдущей purged validation минимизацией worst normal/delayed Brier. Candidate threshold и cooldown state строятся по нормальной прошлой истории, test labels не используются. Отдельной перекалибровки финальной смеси нет, поэтому probability на свежих датах остаётся ровно Halyk lag1.

Это observation-date age в snapshot/lag simulation, не подтверждённый production feed-health или архив publication timestamps. Следует проверять timestamp и фактическую задержку публикации при боевой интеграции. Никакого поиска дополнительных gate-порогов или новых моделей после этих двух arms.
