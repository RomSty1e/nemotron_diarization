# Nemotron Diarization Lab

Готовая стартовая версия проекта для WAV → вероятности спикеров → собственный postprocessing → RTTM.
Инструкции проверены по официальным страницам 29.09.2026. Полный нейросетевой запуск на GPU в среде подготовки архива НЕ выполнялся. Проверены синтаксис Python/Bash и 6 тестов postprocessing/RTTM.

## 1. Что понадобится

- Linux; команды ниже рассчитаны на Ubuntu/Debian. На Windows запускай проект в WSL2 с работающим доступом к NVIDIA GPU.
- NVIDIA GPU и совместимый драйвер. Проверь `nvidia-smi` на той машине/Slurm-узле, где будет inference.
- Git, curl, unzip, ffmpeg, libsndfile и инструменты сборки.
- `uv`. Скрипт создаёт отдельную среду в `third_party/Speech/.venv`, использует Python 3.13 и зависимости из `uv.lock` NVIDIA. Вручную устанавливать torch поверх этой среды не нужно.
- Доступ к GitHub, PyPI/PyTorch package indexes и Hugging Face при установке и загрузке весов.

Системные пакеты (нужны права администратора):

```bash
sudo apt-get update
sudo apt-get install -y git curl unzip ffmpeg libsndfile1 build-essential pkg-config
```

На кластере без sudo попроси администратора предоставить недостающие пакеты или используй доступные environment modules. Наш установочный скрипт не запускает sudo и не меняет системный драйвер.

Установка uv, если его ещё нет:

```bash
curl -LsSf https://astral.sh/uv/install.sh -o /tmp/install-uv.sh
sh /tmp/install-uv.sh
source "$HOME/.local/bin/env"
uv --version
```

## 2. Распаковать и установить

```bash
unzip nemotron_diarization.zip
cd nemotron_diarization
bash scripts/setup_env.sh cu12
source third_party/Speech/.venv/bin/activate
```

Выбери ровно один вариант: `cu12` или `cu13`. Первый приведён как пример для CUDA 12; выбор зависит от совместимости драйвера и сборки PyTorch. Значение CUDA Version в `nvidia-smi` — возможность драйвера, а не версия установленного Python-пакета torch. Не обновляй драйвер на кластере самостоятельно.

Скрипт инициализирует Git в распакованной папке, добавляет NVIDIA Speech как submodule, устанавливает ASR collection в editable mode и наш проект в ту же среду. Веса он не скачивает. При первой установке берётся текущая ветка `main`; это **не заранее закреплённый и протестированный commit**. SHA установленного Speech записывается в `third_party/Speech.commit`. Повторный запуск не обновляет существующий submodule автоматически.

Зафиксируй рабочую версию после успешного запуска:

```bash
git add .gitmodules third_party/Speech configs inference scripts tests pyproject.toml README.md .gitignore
git commit -m "Add Nemotron diarization project and pin Speech submodule"
```

Если нужна конкретная ревизия NVIDIA при установке:

```bash
NEMO_REVISION=<commit_sha> bash scripts/setup_env.sh cu12
```

Не используй `uv sync` из корня нашего проекта: средой управляет проект `third_party/Speech`. Если после повторного `uv sync` исчезла наша установка, снова запусти `setup_env.sh` — он установит её обратно.

## 3. Скачать checkpoint

```bash
bash scripts/download_model.sh
```

Ожидаемый путь:

`checkpoints/nemotron_3_diarization/Nemotron-3-Diarization.nemo`

Если Hugging Face требует авторизацию / возвращает 401 или 403:

```bash
source third_party/Speech/.venv/bin/activate
hf auth login
bash scripts/download_model.sh
```

Для фиксации ревизии весов можно передать `MODEL_REVISION=<hf_commit_sha>`. Сами веса и NVIDIA Speech не включены в ZIP. Если `.nemo` уже скачан, скопируй его по указанному пути либо поменяй `model.path` в YAML. Несуществующий локальный `.nemo` вызывает понятную ошибку, а не попытку поиска такого имени на HF.

## 4. Проверить окружение

```bash
python -m scripts.check_env
```

Проверь `CUDA available: True` и путь NeMo, ведущий в `third_party/Speech/nemo`. Это означает, что правки в исходниках будут подхватываться новым процессом Python. На Slurm проверяй внутри выделения с GPU. Для одного запуска используется один GPU; DDP и multi-node здесь не нужны.

## 5. Подготовить аудио и запустить

CLI проверяет mono / 16000 Hz и не выполняет неявного resampling.

```bash
ffmpeg -i /path/to/input.wav -ac 1 -ar 16000 -c:a pcm_s16le /path/to/input_16k.wav

python -m scripts.diarize \
  --config configs/offline.yaml \
  --audio /path/to/input_16k.wav \
  --output-dir outputs/test
```

Все команды запускай из корня проекта. Относительные пути в YAML отсчитываются от текущей рабочей директории. При пробелах в имени WAV укажи `--recording-id session_001`.

Выходные файлы:

- `input_16k.probs.pt`: CPU tensor `[T, S]`, шаг времени, длительность записи, ID, конфиг и метаданные.
- `input_16k.rttm`: сегменты с сохранением одновременной активности нескольких спикеров.
- `input_16k.run.json`: параметры запуска и записанный при установке commit NeMo.

Существующие результаты защищены от случайной перезаписи; для замены добавь `--overwrite`.
Выбрать GPU: `CUDA_VISIBLE_DEVICES=1 python -m scripts.diarize ...`; внутри процесса он станет `cuda:0`. Для диагностики можно добавить `--device cpu`, но полный inference может быть медленным.

## 6. Изменить threshold без повторного inference

```bash
python -m scripts.reprocess \
  --probs outputs/test/input_16k.probs.pt \
  --output outputs/test/input_16k_thr045.rttm \
  --threshold 0.45 \
  --min-speech-duration 0.10 \
  --max-gap-duration 0.10
```

Postprocessing: независимый threshold каждого speaker channel → склейка коротких пауз → фильтр коротких сегментов → обрезка по длительности WAV. Минимальная длительность округляется вверх по сетке кадров, допустимая пауза — вниз. Overlap сохраняется. Speaker IDs локальны для записи; одинаковый `speaker_0` в разных WAV не означает одного человека.

Просмотр вероятностей:

```python
import torch
x = torch.load('outputs/test/input_16k.probs.pt', map_location='cpu', weights_only=True)
print(x['probs'].shape, x['frame_duration'])
```

## 7. Где менять код

| Путь | Назначение |
|---|---|
| `inference/model.py` | Загрузка checkpoint, параметры chunk/cache |
| `inference/offline.py` | Получение вероятностей через NeMo |
| `inference/audio.py` | Проверка входного аудио |
| `inference/postprocess.py` | Вероятности → сегменты |
| `inference/rttm.py` | Запись RTTM |
| `scripts/diarize.py` | Один WAV → все результаты |
| `scripts/reprocess.py` | Повторная обработка `.probs.pt` |
| `third_party/Speech` | Исходники NVIDIA после установки |

`offline.yaml` означает обработку готовой записи внутренним chunked inference NeMo. Пресеты `low_latency.yaml`, `very_low_latency.yaml`, `ultra_low_latency.yaml` можно подставить в тот же CLI для сравнения. Это НЕ реализация live-микрофона или собственного `step()` API: cache/FIFO ведёт NeMo внутри `diarize()`. NVIDIA postprocessing также выполняется внутри этого вызова; его сегменты отбрасываются, собственный RTTM строится из tensor outputs.

Chunk/cache параметры заданы в 80-ms encoder frames. Шаг выходных вероятностей определяется по модели; не считай, что он обязательно равен 80 ms. В этой версии нет настройки изменения output resolution после загрузки.

Правки NVIDIA делай в submodule и сохраняй в своём fork/commit: главный репозиторий хранит только ссылку на commit submodule, а не его незакоммиченный diff.

## 8. Типичные проблемы

- `uv: command not found`: открой новый shell или активируй путь uv, как выше.
- `CUDA unavailable`: проверь выделение GPU, `nvidia-smi`, активированную среду и `torch.version.cuda`.
- Ошибка совместимости драйвера: согласуй CUDA extra со стеком кластера. Не исправляй её случайной переустановкой torch внутри locked-среды.
- Ошибка `uv sync --locked`: сохрани текст ошибки; выбранный commit/его lockfile или доступ к пакетным индексам может требовать отдельной настройки. Не удаляй lockfile автоматически.
- `ModuleNotFoundError: nemo`: активируй `third_party/Speech/.venv`; запусти `setup_env.sh` и `check_env`.
- Missing/unexpected keys при restore: в YAML стоит `strict: false`, как в примере NVIDIA. Просмотри предупреждения; для строгой диагностики поставь `strict: true`. Несовместимый checkpoint не следует считать успешно проверенным только потому, что его удалось загрузить.
- Неверный sample rate / stereo: конвертируй через ffmpeg.
- OOM: batch уже равен 1; попробуй меньший chunk preset или GPU с большей памятью. Пресет меняет условия inference и может изменить качество. Нарезка записи на независимые WAV обнуляет speaker identity между частями.
- Ошибка проверки длины вероятностей: сохрани лог, shape и `frame_duration`; не подгоняй временную сетку простым делением длительности WAV на число кадров.

## 9. Локальные тесты

```bash
python -m unittest discover -s tests -v
```

Тесты не скачивают веса и не требуют GPU. Они не подтверждают качество модели или корректность полной интеграции на твоём окружении. После установки первый smoke test — короткий WAV с двумя спикерами и просмотр полученного RTTM.

## Официальные источники

- https://huggingface.co/nvidia/Nemotron-3-Diarization — checkpoint, API и пресеты.
- https://github.com/NVIDIA-NeMo/Speech/blob/main/docs/source/starthere/install.rst — установка.
- https://github.com/NVIDIA-NeMo/Speech/blob/main/nemo/collections/asr/models/sortformer_diar_models.py — tensor outputs и временная сетка.

NVIDIA Speech и checkpoint распространяются по своим лицензиям; архив содержит только обвязку проекта.
