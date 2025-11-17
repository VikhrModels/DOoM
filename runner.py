import yaml
from typing import Dict, Any, Optional
from src.equality_checker import DoomSlayer
from src.leaderboard import Leaderboard
import argparse
from pathlib import Path
import sys
import os
import shutil


def main() -> None:
    """
    Основная функция приложения для оценки языковых моделей на математических и физических задачах.

    Обрабатывает аргументы командной строки, запускает оценку моделей на выбранных датасетах
    и выводит результаты в виде отформатированной таблицы.
    """
    # Установка кодировки вывода в UTF-8
    if sys.platform == "win32":
        os.system("chcp 65001")  # Установка кодировки UTF-8 для Windows консоли

    parser = argparse.ArgumentParser(
        description="Оценка языковых моделей на математических и физических задачах",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python runner.py                        # Запустить оценку на всех датасетах (по умолчанию)
  python runner.py --dataset russianmath  # Запустить только на датасете RussianMath
  python runner.py --dataset demidovich  # Запустить только на датасете MathDemon_Demidovich
  python runner.py --dataset physics      # Запустить только на датасете RussianPhysics
  python runner.py --no-cache             # Игнорировать кэш и переоценить все модели
  python runner.py --max-workers 8        # Использовать 8 параллельных потоков
        """,
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Максимальное количество параллельных потоков (по умолчанию: 4)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/run.yaml",
        help="Путь к файлу конфигурации (по умолчанию: configs/run.yaml)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Игнорировать кэш и переоценить все модели",
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "russianmath", "physics", "demidovich"],
        default="all",
        help="Выбор датасета для оценки: all (все), russianmath, physics (по умолчанию: all)",
    )
    parser.add_argument(
        "--retry-incomplete",
        action="store_true",
        help="Перезапустить оценку моделей с неполными результатами",
    )
    args = parser.parse_args()

    # Загружаем конфиг
    with open(args.config, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    # Если указан --no-cache, отключаем использование кэша
    if args.no_cache:
        cache_dir = Path("results/cache")
        if cache_dir.exists():
            for cache_file in cache_dir.glob("*.json"):
                cache_file.unlink()

    # Создаем equality checker для проверки равенства математических выражений
    equality_checker = DoomSlayer()


    # Создаем и инициализируем лидерборд
    leaderboard = Leaderboard(args.config, max_workers=args.max_workers, retry_incomplete=args.retry_incomplete)

    # Определяем системные промпты для каждой модели из конфига
    system_prompts: Dict[str, Optional[str]] = {
        model: config.get(model, {}).get("system_prompt")
        for model in config["model_list"]
    }

    # Запуск оценки в зависимости от выбранного датасета
    if args.dataset == "all" or args.dataset == "russianmath":
        print("\nЗапуск оценки на датасете RussianMath")
        leaderboard.evaluate_all_models(system_prompts)

    if args.dataset == "all" or args.dataset == "demidovich":
        print("\nЗапуск оценки на датасете MathDemon_Demidovich")
        leaderboard.evaluate_demidovich_models(system_prompts)

    if args.dataset == "all" or args.dataset == "physics":
        print("\nЗапуск оценки на датасете RussianPhysics")
        leaderboard.evaluate_physics_models(system_prompts)

    # Вычисляем общий скор для моделей (полусумма по обоим датасетам)
    if args.dataset == "all":
        print("\nВычисление общего скора по всем датасетам")
        leaderboard.calculate_combined_scores()

    # Получаем ширину терминала для форматирования вывода
    terminal_width = shutil.get_terminal_size().columns

    # Выводим красивый заголовок лидерборда
    header = " LEADERBOARD "
    padding = "=" * ((terminal_width - len(header)) // 2)
    print(f"\n{padding}{header}{padding}")

    # Генерируем markdown таблицу с результатами
    md = leaderboard.generate_markdown()

    # Форматируем и выводим таблицу в терминал
    lines = md.split("\n")
    table_lines = [line for line in lines if line.startswith("|")]

    if len(table_lines) >= 2:  # Есть заголовок и разделитель
        header_line = table_lines[0]
        separator_line = table_lines[1]
        data_lines = table_lines[2:] if len(table_lines) > 2 else []

        # Анализируем ширину каждого столбца из заголовка
        columns = header_line.split("|")
        columns = [col.strip() for col in columns if col]  # Убираем пустые элементы

        # Находим максимальную ширину для каждого столбца
        column_widths = [len(col) for col in columns]

        # Учитываем ширину данных в каждой строке
        for line in data_lines:
            cells = line.split("|")
            cells = [cell.strip() for cell in cells if cell]
            for i, cell in enumerate(cells):
                if i < len(column_widths):
                    column_widths[i] = max(column_widths[i], len(cell))

        # Форматируем и выводим заголовок
        formatted_header = (
            "| "
            + " | ".join(f"{col:<{column_widths[i]}}" for i, col in enumerate(columns))
            + " |"
        )
        print(f"\n{formatted_header}")

        # Форматируем и выводим разделитель
        formatted_separator = (
            "|-" + "-|-".join("-" * width for width in column_widths) + "-|"
        )
        print(formatted_separator)

        # Форматируем и выводим данные
        for line in data_lines:
            cells = line.split("|")
            cells = [cell.strip() for cell in cells if cell]
            formatted_line = (
                "| "
                + " | ".join(
                    f"{cell:<{column_widths[i]}}" for i, cell in enumerate(cells)
                )
                + " |"
            )
            print(formatted_line)
    else:
        # Если не смогли разобрать таблицу, выводим строки как есть
        for line in table_lines:
            print(line)

    print(f"\nДетальные результаты сохранены в: {leaderboard.output_dir}")


if __name__ == "__main__":
    main()
