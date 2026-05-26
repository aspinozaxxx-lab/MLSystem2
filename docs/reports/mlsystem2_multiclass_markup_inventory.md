# Инвентаризация MLMarkup для multiclass segmentation

Дата проверки: 2026-05-26.

Серверный контур:
- сервер: `ssh gpu-mlserver`;
- MLMarkup: `/data/MLMarkup`;
- prepared images: `/data/mlsystem2/prepared_images/`.

Команды инвентаризации:

```bash
find /data/MLMarkup -maxdepth 3 -type f \( -name "*.geojson" -o -name "*.txt" \) | sort
find /data/MLMarkup -maxdepth 2 -type d | sort
```

## Найденные классы

| class slug | Русское имя | Найденная папка | scenes_file | annotation_file | scenes count | objects count |
|---|---|---|---|---|---:|---:|
| abrasion | Абразия | `/data/MLMarkup/Абразия` | `/data/MLMarkup/Абразия/abrasion.txt` | `/data/MLMarkup/Абразия/abrasion.geojson` | 3 | 11 |
| wind_erosion | Ветровая эрозия | `/data/MLMarkup/Ветровая эрозия` | `/data/MLMarkup/Ветровая эрозия/wind_erosion.txt` | `/data/MLMarkup/Ветровая эрозия/wind_erosion.geojson` | 2 | 21 |
| water_erosion | Водная эрозия | `/data/MLMarkup/Водная эрозия` | `/data/MLMarkup/Водная эрозия/water_erosion.txt` | `/data/MLMarkup/Водная эрозия/water_erosion.geojson` | 1 | 149 |
| deforestation | Вырубки | `/data/MLMarkup/Вырубки` | `/data/MLMarkup/Вырубки/deforestation.txt` | `/data/MLMarkup/Вырубки/deforestation.geojson` | 35 | 534 |
| fire | Гари | `/data/MLMarkup/Гари` | `/data/MLMarkup/Гари/burnt_forests.txt` | `/data/MLMarkup/Гари/burnt_forests.geojson` | 7 | 79 |
| waterlogging | Заболачивание | `/data/MLMarkup/Заболачивание` | `/data/MLMarkup/Заболачивание/swampings.txt` | `/data/MLMarkup/Заболачивание/swampings.geojson` | 0 | 62 |
| salinization | Засоления | `/data/MLMarkup/Засоления` | `/data/MLMarkup/Засоления/salty.txt` | `/data/MLMarkup/Засоления/salty.geojson` | 4 | 156 |
| quarries | Карьеры | `/data/MLMarkup/Карьеры` | `/data/MLMarkup/Карьеры/careers.txt` | `/data/MLMarkup/Карьеры/careers.geojson` | 7 | 94 |
| landslide_scree | Обвально-оползневые и осыпные | `/data/MLMarkup/Обвально-оползневые и осыпные` | `/data/MLMarkup/Обвально-оползневые и осыпные/landslides.txt` | `/data/MLMarkup/Обвально-оползневые и осыпные/landslides.geojson` | 2 | 201 |
| lakes | Озера | `/data/MLMarkup/Озера` | `/data/MLMarkup/Озера/lakes.txt` | `/data/MLMarkup/Озера/lakes.geojson` | 124 | 202 |
| desertification | Опустынивание | `/data/MLMarkup/Опустынивание` | `/data/MLMarkup/Опустынивание/desertification.txt` | `/data/MLMarkup/Опустынивание/desertification.geojson` | 12 | 241 |
| arable_lands | Пашни | `/data/MLMarkup/Пашни` | `/data/MLMarkup/Пашни/areas_of_used_arable_land.txt` | `/data/MLMarkup/Пашни/areas_of_used_arable_land.geojson` | 3 | 30 |
| rivers | Реки | `/data/MLMarkup/Реки` | `/data/MLMarkup/Реки/rivers.txt` | `/data/MLMarkup/Реки/rivers.geojson` | 0 | 12 |

## Выводы

Все 13 запрошенных классов найдены. Blocker по отсутствующим папкам или файлам отсутствует.

Особенности данных:
- у классов `waterlogging` и `rivers` файлы списка сцен существуют, но после удаления пустых строк и комментариев дают `0` сцен;
- smoke train можно запускать с текущим config, потому общий объединенный список сцен непустой, но полноценное обучение требует уточнить или заполнить списки сцен для этих двух классов;
- пути в примере `configs/multiclass.example.server.yaml` указаны по фактической инвентаризации, без выдуманных имен файлов.
