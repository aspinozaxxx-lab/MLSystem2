# Frontend MLSystem2

Стек: статический SPA на HTML/CSS/JavaScript без Node.js. Это сделано намеренно: текущий CI проекта
Python-only, поэтому сборка выполняется командой:

```bash
python frontend/build.py
```

Сборка копирует `frontend/src` в `frontend/dist`. В production `frontend/dist` отдается статикой,
а все запросы `/api/v1/*` проксируются в `training_ui_api`.

Фронт не обращается к Postgres и не хранит defaults в constants: датасеты, модели, шаблоны, очереди и
результаты берутся только из FastAPI. Датасеты MLMarkup отображаются как варианты `Класс\вариант`; на
странице результатов класс показывает вложенные ссылки на варианты, сам класс не является выбираемым
датасетом.
