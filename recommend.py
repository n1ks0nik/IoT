def recommend_actions(pred_abv: float, target: float, df_run) -> list[str]:
    delta = target - pred_abv
    tips = []
    # 1. Температура
    T_now   = df_run["temperature"].iloc[-1]
    if delta > 0.2 and T_now < 25:
        tips.append("⏫ Поднять температуру на 1–1.5 °C "
                    "— ускорит сбраживание и повысит ABV.")
    if delta < -0.2 and T_now > 18:
        tips.append("⏬ Понизить температуру на ~1 °C "
                    "— замедлит дрожжи и снизит ABV.")
    # 2. Время
    if delta > 0.1:
        tips.append("⏳ Увеличить время активного брожения ещё на 12 ч.")
    # 3. Питательные вещества
    if delta > 0.15 and df_run["sg"].iloc[-1] > 1.020:
        tips.append("🍯 Добавить дрожжевое питание (DAP) "
                    "для более полного сбраживания.")
    return tips or ["Текущий процесс уже должен дать целевой ABV."]
