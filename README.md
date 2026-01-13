## Симуляция VO с Loop Closure (для дронов) (Classic + SuperGlue)

Этот проект содержит симулятор, который:
- вырезает движущееся окно камеры из большого аэроизображения,
- стабилизирует картинку по известному (oracle) углу рысканья (yaw),
- сопоставляет соседние кадры в стабилизированном виде,
- оценивает сдвиг (только трансляция) с помощью RANSAC,
- при необходимости выполняет замыкание петель (loop closure) через сопоставление с ключевыми кадрами и распределение дрейфа по «хвосту» траектории,
- считает метрики и сохраняет изображения траекторий.

Включённые методы:
- SuperPoint + SuperGlue (опционально; требует внешний репозиторий и веса)
- ORB (классический, с Loop Closure)
- SIFT (классический, с Loop Closure)
- AKAZE (классический, с Loop Colusre)
- VO-only (SuperPoint + SuperGlue, без замыкания петель) для сравнения дрейфа

Важное замечание про большие изображения:
- Файл большого изображения `.tif` не входит в репозиторий. Добавьте свой файл (например, `big.tif`) самостоятельно и передайте путь через `--image`.
- Вы можете положить `big.tif` рядом со скриптами (в `public_release/` или запустить из корня и указать относительный путь).
- В `public_release/.gitignore` добавлено правило игнорирования `*.tif`. Если хотите приложить небольшой пример, поместите его, например, в `public_release/samples/` и скорректируйте `.gitignore`.

### Структура папок
- `methods/` — исполняемые скрипты
  - `superpoint_simulation_aerial_with_loopclosure_posegraph_from_trajectory.py` (вариант SuperGlue)
  - `superpoint_simulation_aerial_vo_only_from_trajectory.py` (VO-only)
  - `superpoint_simulation_aerial_orb_with_loopclosure_posegraph_from_trajectory.py`
  - `superpoint_simulation_aerial_sift_with_loopclosure_posegraph_from_trajectory.py`
  - `superpoint_simulation_aerial_akaze_with_loopclosure_posegraph_from_trajectory.py`
  - `run_all_methods_benchmark.py` (запускает несколько методов и формирует сводку)
- `metrics.py` — метрики (ATE, RPE, фичи/инлиеры и т.д.)
- `magicpoint/supereye.py` — фронтенд SuperPoint (для варианта SuperGlue)

### Требования

```bash
pip install -r requirements.txt
```

Примечания:
- Для SIFT нужен `opencv-contrib-python`.
- Для SuperGlue нужны PyTorch и внешний репозиторий (см. ниже).

### Необязательно (SuperGlue)
1) Клонируйте репозиторий SuperGlue рядом с `methods/`:
   - `git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git SuperGluePretrainedNetwork-master`
2) Поместите веса SuperPoint в `magicpoint/superpoint_v1.pth`.

Если пропустите этот шаг, вы всё равно можете запускать классические методы (ORB/SIFT/AKAZE) и общий бенчмарк с `--exclude-sg`.

### Входные данные
- Большое аэроизображение: один файл `.tif` или `.png`.
- JSON с траекторией: структура `{"trajectory": [{"x": ..., "y": ..., "yaw": ...}, ...], "meta": {"image_size": [W, H]}}`
  - Если есть `meta.image_size`, координаты будут масштабированы под размеры вашего изображения. Иначе координаты считаются в пикселях текущего изображения.
  - `yaw` в радианах.

### Запуск одиночных методов

SuperGlue (замыкание петель включено):
```bash
python methods/superpoint_simulation_aerial_with_loopclosure_posegraph_from_trajectory.py --image big.tif --trajectory square_loop.json --loops 1
```

VO-only (SuperGlue, без замыкания петель):
```bash
python methods/superpoint_simulation_aerial_vo_only_from_trajectory.py --image big.tif --trajectory square_loop.json --loops 1
```

ORB (классика, с замыканием петель):
```bash
python methods/superpoint_simulation_aerial_orb_with_loopclosure_posegraph_from_trajectory.py --image big.tif --trajectory square_loop.json --loops 1
```

SIFT (классика, с замыканием петель):
```bash
python methods/superpoint_simulation_aerial_sift_with_loopclosure_posegraph_from_trajectory.py --image big.tif --trajectory square_loop.json --loops 1
```

AKAZE (классика, с замыканием петель):
```bash
python methods/superpoint_simulation_aerial_akaze_with_loopclosure_posegraph_from_trajectory.py --image big.tif --trajectory square_loop.json --loops 1
```

Общие флаги:
- `--window 300` — размер окна камеры
- `--max-kp 500` — максимум ключевых точек
- `--yaw-use prev|cur|mid` — какую оценку yaw использовать (по умолчанию авто-калибровка)
- `--yaw-sign 1.0` — знак yaw (если ваша конвенция отличается)
- `--log-every 200` — печать прогресса каждые N кадров
- `--no-loop-closure` — выключить замыкание петель в соответствующих скриптах
- Выводы: каждый метод сохраняет карту траектории и оверлей (имена файлов автоматически формируются по методу, если путь не задан)

### Массовый запуск (несколько методов)

Все методы с замыканием петель (SuperGlue + классические). Сохраняет по два изображения на метод и суммарный текстовый отчёт:
```bash
python methods/run_all_methods_benchmark.py --image big.tif --trajectory square_loop.json --loops 5
```

Только классика (без нейросетей):
```bash
python methods/run_all_methods_benchmark.py --image big.tif --trajectory square_loop.json --loops 5 --exclude-sg
```

Добавить VO-only:
```bash
python methods/run_all_methods_benchmark.py --image big.tif --trajectory square_loop.json --loops 5 --include-vo-only
```

Выводы:
- Изображения сохраняются в текущую рабочую директорию:
  - `traj_map_sg.png`, `gt_overlay_sg.png`
  - `traj_map_orb.png`, `gt_overlay_orb.png`
  - `traj_map_sift.png`, `gt_overlay_sift.png`
  - `traj_map_akaze.png`, `gt_overlay_akaze.png`
  - `traj_map_vo.png`, `gt_overlay_vo.png` (если указан `--include-vo-only`)
- `metrics_summary.txt` — по одной строке на метод (кадры, ошибки трансляции/вращения, ATE, RPE, фичи, доля инлиеров)
- `method_logs/*.log` — полные логи консоли для каждого запуска

### Советы
- Если появляются ошибки импортов, запускайте из корня репозитория, чтобы скрипты смогли добавить корень проекта в `sys.path`.
- Для SIFT убедитесь, что установлен `opencv-contrib-python`.
- Для SuperGlue убедитесь, что папка внешнего репозитория называется `SuperGluePretrainedNetwork-master`, а веса лежат в `magicpoint/superpoint_v1.pth`.



