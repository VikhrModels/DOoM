# Doom Benchmark

<p align="center">
  <img src="images/Logo.png" alt="Doom Logo" width="300"/>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.13-blue.svg" alt="Python Version"/></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License"/></a>
  <a href="https://huggingface.co/spaces/Vikhrmodels/DOoM-lb"><img src="https://img.shields.io/badge/🤗-HuggingFace%20Space-yellow.svg" alt="HuggingFace Space"/></a>
  <a href="https://openrouter.ai"><img src="https://img.shields.io/badge/OpenRouter-Supported-success.svg" alt="OpenRouter Support"/></a>
</p>

Doom - бенчмарк для оценки качества языковых моделей на математических и физических задачах на русском языке.

## 📖 О проекте

Doom - это инструмент для тестирования и оценки способности языковых моделей (LLM) решать задачи по математике и физике. Бенчмарк позволяет:

- Измерять точность решения математических задач
- Оценивать понимание физических концепций и способность решать задачи по физике
- Сравнивать производительность разных моделей на русскоязычном контенте
- Оценивать улучшения в способностях моделей к решению научных задач
- **Интегрировать модели через различные API, включая OpenRouter для доступа к широкому спектру моделей**

Основная часть кодовой базы адаптирована из проекта OpenAI simpleeval.

## 📊 Поддерживаемые датасеты

1. **RussianMath** - разнообразные задачи по математике на русском языке (основной математический датасет)
2. **RussianPhysics** - задачи по физике на русском языке (основной физический датасет)
3. **Специализированные задачи** - многозадачные проблемы с контекстом до **200 000+ токенов**

## 🚀 Запуск

### Установка зависимостей

Рекомендуется использовать `uv` и Python 3.13 для лучшей совместимости и производительности:

```bash
# Установка uv (если еще не установлен)
# Windows (PowerShell)
pip install uv

# Создание виртуального окружения
uv venv venv -p 3.13

# Активация окружения
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

# Установка зависимостей с помощью uv
uv pip install -r requirements.txt
```

Альтернативный вариант (стандартный pip):
```bash
pip install -r requirements.txt
```

### Базовый запуск (все датасеты)

```bash
python runner.py
```

### Запуск с конфигурацией OpenRouter

```bash
python runner.py --config configs/run.yaml --model anthropic/claude-sonnet-4
```

### Выбор конкретного датасета

```bash
python runner.py --dataset russianmath  # Только датасет RussianMath
python runner.py --dataset physics      # Только датасет RussianPhysics
```

### Другие параметры

```bash
python runner.py --no-cache       # Игнорировать кэш и повторно выполнить оценку
python runner.py --max-workers 8  # Установить количество параллельных обработчиков
python runner.py --config path/to/config.yaml  # Указать альтернативный конфиг
```

### Справка по параметрам

```bash
python runner.py --help
```

## ⚙️ Конфигурация

Настройка выполняется через файлы YAML в директории `configs/`:

```yaml
configs/run.yaml  # Основной конфигурационный файл
```

### Поддерживаемые API типы:
- `openai` - для OpenAI и совместимых API 
- `gigachat` - для моделей GigaChat
- `openrouter` - для доступа к моделям через OpenRouter (новое)

### Пример конфигурации для моделей через OpenRouter

```yaml
model_list:
  - anthropic/claude-sonnet-4  # Claude Sonnet 4 с 200K контекстом

anthropic/claude-sonnet-4:
  model_name: "anthropic/claude-sonnet-4"  # Отображаемое имя
  endpoints:
    - api_base: "https://openrouter.ai/api/v1"  # Обязательный URL
      api_key: "your_openrouter_api_key"       # Ключ с openrouter.ai
      referer_url: "https://your-project.edu"  # Ваш домен (обязательно)
      app_title: "DOoM Benchmark"              # Название приложения (опционально)
  api_type: openrouter        # Тип API: openrouter
  parallel: 2                 # Рекомендуется 2 параллельных запроса
  system_prompt: |            # Промпт для научных задач
    Ты - экспертный решатель физико-математических задач. Строго следуй:
    1. Пошаговый анализ проблемы
    2. Точные вычисления с промежуточными результатами
    3. Финальный ответ в формате: Ответ: $RESULT
    4. Единицы измерений используй только если они указаны в условии
  max_tokens: 190000          # Поддержка полного контекста модели
  temperature: 0.0            # Для детерминированных результатов
  request_delay: 0.5          # Задержка между запросами (рекомендовано)
```

### Полный пример конфига
```yaml
model_list:
  - gpt-4o-mini
  - gigachat-pro
  - anthropic/claude-sonnet-4

# Общие настройки
debug: false
num_examples: 50

gpt-4o-mini:
  model_name: "GPT-4o Mini"
  endpoints:
    - api_base: "https://api.openai.com/v1"
      api_key: "your_openai_api_key"
  api_type: openai
  # ... остальные параметры ...

anthropic/claude-sonnet-4:
  # ... параметры Claude Sonnet 4 ...
```

**Требования для OpenRouter:**
- Обязательный параметр `referer_url` с валидным URL проекта
- Регистрация на [OpenRouter.ai](https://openrouter.ai) для получения API ключа
- Формат имени модели: `провайдер/имя-модели` (напр. `anthropic/claude-sonnet-4`)

**Описание параметров конфигурации:**

*   `model_list`: Список моделей для оценки.
*   `[model_name]`: Блок конфигурации для конкретной модели. Имя блока должно совпадать с именем в `model_list`.
    *   `model_name`: Имя модели (может отличаться от ключа блока, например, для локальных моделей).
    *   `endpoints`: Список эндпоинтов API.
        *   `api_base` / `base_url`: URL API.
        *   `api_key` / `credentials`: Ключ API или учетные данные (зависит от `api_type`).
        *   `referer_url`: Обязательный для OpenRouter URL вашего проекта.
        *   `app_title`: Название приложения для статистики OpenRouter.
    *   `api_type`: Тип API (`openai`, `gigachat`, `openrouter`).
    *   `parallel`: Количество параллельных запросов к API для этой модели.
    *   `system_prompt`: Системный промпт для модели.
    *   `max_tokens`: Максимальное количество токенов в ответе.
    *   `num_examples` (Опционально): Количество примеров для оценки этой модели. Переопределяет глобальное значение `num_examples`. **По умолчанию используются все доступные примеры из датасета.**
*   `num_examples` (Глобально, Опционально): Количество примеров для оценки для всех моделей, если не переопределено в блоке модели. **По умолчанию используются все доступные примеры из датасета.**
*   `debug` (Глобально, Опционально): Включить режим отладки для вывода дополнительной информации.

## 📝 Результаты тестирования

После запуска оценки автоматически будет сгенерирована таблица лидеров.
Она сохраняется в файле `results/leaderboard.md`.

**Новые возможности:**
- Поддержка моделей с длинным контекстом (до 200 000 токенов)
- Интеграция результатов моделей OpenRouter в лидерборд
- Автоматическое определение потребления токенов для расчета стоимости

Детальные результаты по каждой модели доступны в директории `results/details/`.

### Публикация результатов

Вы можете опубликовать результаты тестирования своей модели в общем лидерборде:

1. Клонируйте репозиторий и запустите тесты вашей модели
2. Загрузите результаты через [HuggingFace Space](https://huggingface.co/spaces/Vikhrmodels/DOoM-lb)
3. Дождитесь проверки и добавления результатов в лидерборд

Формат результатов для публикации в JSON формате:
```json
{
  "score": 0.586,
  "math_score": 0.8,
  "physics_score": 0.373,
  "total_tokens": 1394299,
  "cost_estimation": 4.28, 
  "evaluation_time": 4533.2,
  "system_prompt": "Ты - экспертный решатель научных задач...",
  "model_provider": "OpenRouter"
}
```

## 🧪 Тестирование моделей через OpenRouter

Чтобы протестировать модели через OpenRouter API:

1. Получите API ключ на [OpenRouter Keys](https://openrouter.ai/keys)
2. Добавьте конфигурацию модели в `configs/run.yaml` (см. пример выше)
3. Укажите модель в формате `provider/model-name` (напр. `anthropic/claude-sonnet-4`)
4. Запустите бенчмарк:
```bash
python runner.py --config configs/run.yaml
```

**Поддерживаемые модели:**  
Claude 3.5 Sonnet, Claude 3 Opus, LLaMA 3 70B, Google Gemini Pro, Command R+ и другие, доступные через OpenRouter.

#### Преимущества использования OpenRouter:
- Доступ к сотням моделей через единый API
- Автоматическое вычисление стоимости вызовов
- Поддержка длинного контекста (до 200K+ токенов)
- Интеграция в существующий рабочий процесс без изменений кода

Подробные инструкции по хостингу моделей через VLLM и их тестированию на бенчмарке доступны в файле [Instruction.md](Instruction.md).

## 📚 Структура проекта

- `/configs` - конфигурационные файлы
- `/src` - исходный код бенчмарка (включая поддержку OpenRouter)
- `/results` - результаты тестирования
  - `/results/details` - подробные результаты по каждой модели
  - `/results/cache` - кэш результатов для ускорения повторных запусков
- `/images` - графические ресурсы проекта

## 🤗 Лидерборд

Текущий лидерборд с результатами тестирования различных моделей доступен на [HuggingFace Space](https://huggingface.co/spaces/Vikhrmodels/DOoM-lb).  
**Новое:** добавлен раздел для моделей OpenRouter с указанием стоимости вычислений.

## 📄 Лицензия

Проект распространяется под лицензией Apache 2.0. См. файл LICENSE для получения дополнительной информации.
