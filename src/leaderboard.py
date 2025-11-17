import yaml
from typing import List, Dict, Any, Set, Optional, Tuple
import time
from pathlib import Path
import json
from datetime import datetime
from src.equality_checker import DoomSlayer
from src.sampler import OaiSampler
from src.mat_boy import *
from src.types import SingleEvalResult
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from tqdm.auto import tqdm
import hashlib
import signal
import sys


class Leaderboard:
    """
    Класс для оценки и сравнения языковых моделей на математических задачах.

    Поддерживает кэширование результатов, параллельную оценку моделей,
    генерацию отчетов и составление таблицы результатов.
    """

    def __init__(
        self, config_path: str, output_dir: str = "results", max_workers: int = 4, retry_incomplete: bool = False
    ) -> None:
        """
        Инициализирует лидерборд для оценки моделей.

        Args:
            config_path: Путь к конфигурационному файлу YAML
            output_dir: Директория для сохранения результатов и кэша
            max_workers: Максимальное количество параллельных потоков
            retry_incomplete: Перезапускать оценку моделей с неполными результатами
        """
        self.config_path: str = config_path
        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers: int = max_workers
        self.retry_incomplete: bool = retry_incomplete

        self.details_dir: Path = self.output_dir / "details"
        self.details_dir.mkdir(exist_ok=True)
        self.cache_dir: Path = self.output_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)

        with open(config_path, "r", encoding="utf-8") as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)
        self.model_links: Dict[str, str] = self.config.get("model_links", {})
        self.equality_checker: DoomSlayer = DoomSlayer()
        self.results_file: Path = self.output_dir / "leaderboard_results.json"
        self.results: Dict[str, Dict[str, Any]] = self._load_results()

    # МЕТОДЫ РАБОТЫ С КЭШЕМ

    def _get_cache_key(self, model_name: str, system_prompt: Optional[str]) -> str:
        """
        Генерирует ключ кэша на основе модели и промпта.

        Args:
            model_name: Название модели
            system_prompt: Системный промпт для модели

        Returns:
            MD5-хеш для уникальной идентификации результатов
        """
        safe_model_name = model_name.replace("/", "_")
        cache_data = {
            "model_name": safe_model_name,
            "system_prompt": system_prompt,
            "num_examples": self.config.get("num_examples"),
            "temperature": self.config.get("temperature"),
            "max_tokens": self.config.get("max_tokens"),
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Получает результат из кэша по ключу.

        Args:
            cache_key: Ключ для поиска в кэше

        Returns:
            Результаты оценки модели или None, если кэш не найден
        """
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _save_to_cache(self, cache_key: str, result: Dict[str, Any]) -> None:
        """
        Сохраняет результат в кэш.

        Args:
            cache_key: Ключ для сохранения в кэше
            result: Результаты оценки модели
        """
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, "w") as f:
            json.dump(result, f, indent=2)

    def _load_results(self) -> Dict[str, Dict[str, Any]]:
        """
        Загружает существующие результаты из кэша и основного файла результатов.

        Returns:
            Объединенный словарь результатов оценки всех моделей
        """
        results = {}

        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.json"):
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_result = json.load(f)
                    model_name = cached_result["model_name"]
                    timestamp = cached_result["timestamp"]
                    results[f"{model_name}_{timestamp}"] = cached_result

        if self.results_file.exists():
            with open(self.results_file, "r", encoding="utf-8") as f:
                file_results = json.load(f)
                results.update(file_results)

        return results

    def _save_results(self) -> None:
        """
        Сохраняет все результаты в основной файл.
        """
        with open(self.results_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

    def _get_measured_models(self) -> Set[str]:
        """
        Получает список уже измеренных моделей из кэша.

        Returns:
            Множество имен моделей, для которых есть кэшированные результаты
        """
        measured_models = set()
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.json"):
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                    measured_models.add(cached_data["model_name"])
        return measured_models

    # МЕТОДЫ РАБОТЫ С ОТЧЕТАМИ

    def _save_detailed_results(
        self,
        model_name: str,
        results: List[SingleEvalResult],
        timestamp: str,
        dataset: Optional[str] = None,
    ) -> Path:
        """
        Сохраняет детальные результаты для модели с указанием датасета.

        Args:
            model_name: Название модели
            results: Список результатов оценки для каждого примера
            timestamp: Временная метка для уникальной идентификации
            dataset: Название датасета (опционально)

        Returns:
            Путь к созданному markdown-файлу с отчетом
        """
        safe_model_name = model_name.replace("/", "_")
        model_dir = self.details_dir / safe_model_name
        model_dir.mkdir(exist_ok=True)

        file_suffix = f"_{dataset}" if dataset else ""

        details_file = model_dir / f"details_{timestamp}{file_suffix}.json"
        with open(details_file, "w", encoding="utf-8") as f:
            json.dump(
                results, f, indent=2, default=lambda x: x.__dict__, ensure_ascii=False
            )

        markdown_file = model_dir / f"details_{timestamp}{file_suffix}.md"
        markdown_content = self._generate_markdown_report(
            model_name, results, timestamp, dataset
        )
        with open(markdown_file, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        if dataset == "RussianPhysics":
            math_report = model_dir / f"details_{timestamp}.md"
            if math_report.exists():
                self._combine_detailed_reports(model_name, timestamp)

        return markdown_file

    def _generate_markdown_report(
        self,
        model_name: str,
        results: List[SingleEvalResult],
        timestamp: str,
        dataset: Optional[str] = None,
    ) -> str:
        """
        Генерирует markdown-отчет с детальными результатами для модели.

        Args:
            model_name: Название модели
            results: Список результатов оценки для каждого примера
            timestamp: Временная метка создания отчета
            dataset: Название датасета (опционально)

        Returns:
            Строка с содержимым markdown-отчета
        """
        dataset_info = f" - {dataset}" if dataset else ""

        md = f"# Detailed Results for {model_name}{dataset_info}\n\n"
        md += f"Timestamp: {timestamp}\n\n"

        correct_count = sum(1 for r in results if hasattr(r, "score") and r.score == 1)
        total_count = len(results)
        overall_score = correct_count / total_count if total_count > 0 else 0

        md += "## Summary\n\n"
        md += f"- **Dataset**: {dataset or 'RussianMath'}\n"
        md += f"- **Total examples**: {total_count}\n"
        md += f"- **Correct answers**: {correct_count}\n"
        md += f"- **Score**: {overall_score:.3f}\n\n"

        md += "---\n\n"

        for i, result in enumerate(results, 1):
            md += f"## Example {i}\n\n"

            if hasattr(result, "convo") and result.convo:
                for message in result.convo:
                    if message.get("role") == "user":
                        md += f"### Task\n{message.get('content', '')}\n\n"
                    elif message.get("role") == "assistant":
                        md += f"### Model Response\n{message.get('content', '')}\n\n"

            if hasattr(result, "correct_answer") and result.correct_answer:
                md += f"### Correct Answer\n{result.correct_answer}\n\n"

            if (
                hasattr(result, "extracted_answer")
                and result.extracted_answer is not None
            ):
                md += f"### Extracted Answer\n{result.extracted_answer}\n\n"

            if hasattr(result, "score") and result.score is not None:
                md += f"### Score\n{result.score}\n\n"

            if hasattr(result, "tokens"):
                md += f"### Tokens Used\n{result.tokens}\n\n"

            md += "---\n\n"

        return md

    def _combine_detailed_reports(
        self,
        model_name: str,
        timestamp_math: Optional[str] = None,
        timestamp_physics: Optional[str] = None,
        timestamp_demidovich: Optional[str] = None
    ) -> Optional[Path]:
        """
        Объединяет отчеты по математике и физике в один комбинированный отчет.

        Args:
            model_name: Название модели
            timestamp_math: Временная метка отчета по математике (опционально)
            timestamp_physics: Временная метка отчета по физике (опционально)

        Returns:
            Путь к созданному комбинированному отчету или None при ошибке
        """
        safe_model_name = model_name.replace("/", "_")
        model_dir = self.details_dir / safe_model_name

        if not model_dir.exists():
            print(f"No detailed reports found for model {model_name}")
            return None

        if timestamp_math and timestamp_physics and timestamp_demidovich:
            math_report = model_dir / f"details_{timestamp_math}.md"
            physics_report = (
                model_dir / f"details_{timestamp_physics}_RussianPhysics.md"
            )
            demidovich_report = model_dir / f"details_{timestamp_demidovich}.md"
        else:
            math_reports = list(model_dir.glob("details_*.md"))
            physics_reports = list(model_dir.glob("details_*_RussianPhysics.md"))
            demidovich_reports =  list(model_dir.glob("details_*_MathDemon_Demidovich.md"))

            if not math_reports or not physics_reports:
                print(f"Missing either math or physics reports for model {model_name}")
                return None

            math_report = sorted(
                math_reports, key=lambda x: x.stat().st_mtime, reverse=True
            )[0]
            physics_report = sorted(
                physics_reports, key=lambda x: x.stat().st_mtime, reverse=True
            )[0]
            demidovich_report = sorted(
                demidovich_reports, key=lambda x: x.stat().st_mtime, reverse=True
            )[0]

            timestamp_math = math_report.stem.split("_")[1]
            timestamp_physics = physics_report.stem.split("_")[1]

        if not math_report.exists() or not physics_report.exists() or not demidovich_report.exists():
            print(f"Could not find some reports for model {model_name}")
            return None

        combined_report = model_dir / f"details_{timestamp_math}_combined.md"

        if combined_report.exists():
            print(
                f"Combined report for {model_name} already exists at {combined_report}"
            )
            return combined_report

        with open(math_report, "r", encoding="utf-8") as f:
            math_content = f.read()

        with open(physics_report, "r", encoding="utf-8") as f:
            physics_content = f.read()

        with open(demidovich_report, "r", encoding="utf-8") as f:
            demidovich_content = f.read()

        math_lines = math_content.split("\n")
        physics_lines = physics_content.split("\n")
        demidovich_lines = demidovich_content.split("\n")

        math_start = 0
        physics_start = 0
        demidovich_start = 0
        math_summary_start = 0
        physics_summary_start = 0
        demidovich_summary_start = 0

        for i, line in enumerate(math_lines):
            if line.startswith("## Summary"):
                math_summary_start = i
            if line.startswith("## Example 1"):
                math_start = i
                break

        for i, line in enumerate(physics_lines):
            if line.startswith("## Summary"):
                physics_summary_start = i
            if line.startswith("## Example 1"):
                physics_start = i
                break

        for i, line in enumerate(demidovich_lines):
            if line.startswith("## Summary"):
                demidovich_summary_start = i
            if line.startswith("## Example 1"):
                demidovich_start = i
                break

        combined_content = []
        for i in range(0, math_summary_start):
            if i == 0:
                combined_content.append(
                    f"# Detailed Results for {model_name} - Combined"
                )
            else:
                combined_content.append(math_lines[i])

        math_summary_lines = [
            line
            for line in math_lines[math_summary_start:math_start]
            if "Score" in line
            or "examples" in line
            or "answers" in line
            or "Dataset" in line
        ]
        physics_summary_lines = [
            line
            for line in physics_lines[physics_summary_start:physics_start]
            if "Score" in line
            or "examples" in line
            or "answers" in line
            or "Dataset" in line
        ]
        demidovich_summary_lines = [
            line
            for line in demidovich_lines[demidovich_summary_start:demidovich_start]
            if "Score" in line
            or "examples" in line
            or "answers" in line
            or "Dataset" in line
        ]

        math_score = next((line for line in math_summary_lines if "Score" in line), "")
        physics_score = next(
            (line for line in physics_summary_lines if "Score" in line), ""
        )
        demidovich_score = next(
            (line for line in demidovich_summary_lines if "Score" in line), ""
        )

        try:
            math_score_value = float(math_score.split(":")[-1].strip())
            physics_score_value = float(physics_score.split(":")[-1].strip())
            demidovich_score_value = float(demidovich_score.split(":")[-1].strip())
            combined_score = (math_score_value + physics_score_value + demidovich_score_value) / 3
            combined_score_line = f"- **Combined Score**: {combined_score:.3f}"
        except (ValueError, IndexError):
            combined_score_line = "- **Combined Score**: N/A"

        combined_content.append("## Summary")
        combined_content.append("")
        combined_content.append(combined_score_line)
        combined_content.append("")
        combined_content.append("### Mathematics")
        for line in math_summary_lines:
            combined_content.append(line)
        combined_content.append("")
        combined_content.append("### Physics")
        for line in physics_summary_lines:
            combined_content.append(line)
        combined_content.append("")
        combined_content.append("---")
        combined_content.append("")
        combined_content.append("### Demidovich")
        for line in demidovich_summary_lines:
            combined_content.append(line)
        combined_content.append("")
        combined_content.append("---")
        combined_content.append("")

        combined_content.append("## Mathematics Examples")
        combined_content.append("")
        combined_content.extend(math_lines[math_start:])

        combined_content.append("\n## Physics Examples")
        combined_content.append("")
        combined_content.extend(physics_lines[physics_start:])

        combined_content.append("## Demidovich Examples")
        combined_content.append("")
        combined_content.extend(demidovich_lines[math_start:])

        with open(combined_report, "w", encoding="utf-8") as f:
            f.write("\n".join(combined_content))

        print(
            f"Successfully created combined report for {model_name} at {combined_report}"
        )
        return combined_report

    def _prepare_combined_reports(self) -> None:
        """
        Подготавливает комбинированные отчеты для всех моделей с обоими датасетами.
        """
        models_with_both_datasets = {}

        for key, result in self.results.items():
            model_name = result["model_name"]
            dataset = result.get("dataset")
            timestamp = result.get("timestamp")

            if model_name not in models_with_both_datasets:
                models_with_both_datasets[model_name] = {"math": None, "physics": None, "demidovich": None}

            if dataset == "RussianMath" or dataset is None:
                if (
                    not models_with_both_datasets[model_name]["math"]
                    or timestamp > models_with_both_datasets[model_name]["math"]
                ):
                    models_with_both_datasets[model_name]["math"] = timestamp
            elif dataset == "RussianPhysics":
                if (
                    not models_with_both_datasets[model_name]["physics"]
                    or timestamp > models_with_both_datasets[model_name]["physics"]
                ):
                    models_with_both_datasets[model_name]["physics"] = timestamp
            elif dataset == "MathDemon_Demidovich":
                if (
                    not models_with_both_datasets[model_name]["demidovich"]
                    or timestamp > models_with_both_datasets[model_name]["demidovich"]
                ):
                    models_with_both_datasets[model_name]["demidovich"] = timestamp

        for model_name, timestamps in models_with_both_datasets.items():
            if timestamps["math"] and timestamps["physics"] and timestamps["demidovich"]:
                self._combine_detailed_reports(
                    model_name=model_name,
                    timestamp_math=timestamps["math"],
                    timestamp_physics=timestamps["physics"],
                    timestamp_demidovich=timestamps["demidovich"],
                )

    # МЕТОДЫ ОЦЕНКИ МОДЕЛЕЙ

    def evaluate_model(
        self, model_name: str, system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Оценивает одну модель на датасете RussianMath.

        Args:
            model_name: Название модели для оценки
            system_prompt: Системный промпт для модели (опционально)

        Returns:
            Словарь с результатами оценки модели
        """
        cache_key = self._get_cache_key(model_name, system_prompt)
        cached_result = self._get_cached_result(cache_key)

        if cached_result is not None:
            if self.config.get("debug"):
                print(f"\nUsing cached result for {model_name}")
            return cached_result

        if self.config.get("debug"):
            print(f"\nEvaluating {model_name}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model_name = model_name.replace("/", "_")

        temp_config = self.config.copy()
        temp_config["model_list"] = [model_name]
        if system_prompt is not None:
            temp_config[model_name]["system_prompt"] = system_prompt

        temp_config_path = self.output_dir / f"temp_config_{safe_model_name}.yaml"
        with open(temp_config_path, "w") as f:
            yaml.dump(temp_config, f)

        try:
            sampler = OaiSampler(str(temp_config_path))
            evaluator = RussianMathEval(
                equality_checker=self.equality_checker,
                num_examples=self.config.get("num_examples", None),
                debug=self.config.get("debug", False),
            )

            start_time = time.time()
            results = evaluator(sampler)
            evaluation_time = time.time() - start_time

            self._save_detailed_results(model_name, results.results, timestamp)

            total_tokens = sum(
                r.tokens for r in results.results if hasattr(r, "tokens")
            )

            model_result = {
                "model_name": model_name,
                "score": results.score,
                "total_tokens": total_tokens,
                "evaluation_time": evaluation_time,
                "system_prompt": system_prompt,
                "timestamp": timestamp,
                "cache_key": cache_key,
            }

            self._save_to_cache(cache_key, model_result)
            self.results[f"{model_name}_{timestamp}"] = model_result
            self._save_results()

            return model_result

        finally:
            temp_config_path.unlink(missing_ok=True)

    def evaluate_model_parallel(
        self, args: Tuple[str, Optional[str]]
    ) -> Dict[str, Any]:
        """
        Оценивает одну модель (для использования в ThreadPoolExecutor).

        Args:
            args: Кортеж (model_name, system_prompt)

        Returns:
            Словарь с результатами оценки модели
        """
        model_name, system_prompt = args
        return self.evaluate_model(model_name, system_prompt)

    def evaluate_all_models(
        self, system_prompts: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Оценивает все модели из конфига параллельно с использованием кэша.

        Args:
            system_prompts: Словарь системных промптов для моделей
                (model_name -> system_prompt)
        """
        if system_prompts is None:
            system_prompts = {}

        measured_models = self._get_measured_models()
        config_models = set(self.config["model_list"])
        new_models = config_models - measured_models

        # Определяем модели, которые нужно перетестировать (модели с неполными результатами)
        models_with_incomplete_results = set()
        if self.retry_incomplete:
            # Получаем модели, у которых отсутствуют результаты для RussianMath
            models_with_math_results = set()
            for key, result in self.results.items():
                if (result.get("dataset") == "RussianMath" or result.get("dataset") is None) and result["model_name"] in config_models:
                    models_with_math_results.add(result["model_name"])
            
            models_with_incomplete_results = config_models - models_with_math_results
            if models_with_incomplete_results:
                print(f"\nFound models with incomplete RussianMath results that will be retested: {', '.join(models_with_incomplete_results)}")

        if new_models:
            print(f"\nFound new models to evaluate: {', '.join(new_models)}")

        for model_name in config_models:
            if model_name in measured_models:
                for cache_file in self.cache_dir.glob("*.json"):
                    with open(cache_file, "r", encoding="utf-8") as f:
                        cached_data = json.load(f)
                        if cached_data["model_name"] == model_name:
                            key = f"{model_name}_{cached_data['timestamp']}"
                            self.results[key] = cached_data
                            break

        # Объединяем новые модели и модели с неполными результатами
        models_to_evaluate = new_models | models_with_incomplete_results

        if models_to_evaluate:
            uncached_args = [
                (model_name, system_prompts.get(model_name))
                for model_name in models_to_evaluate
            ]

            print(f"\nEvaluating {len(uncached_args)} models...")

            def handle_sigint(signum: int, frame: Any) -> None:
                print(
                    "\nGracefully shutting down... Please wait for current evaluations to complete."
                )
                executor.shutdown(wait=True)
                sys.exit(0)

            original_sigint = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, handle_sigint)

            try:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = [
                        executor.submit(self.evaluate_model_parallel, args)
                        for args in uncached_args
                    ]

                    pbar = tqdm(
                        total=len(futures), desc="Evaluating models", leave=True
                    )

                    completed = 0
                    while completed < len(futures):
                        for i, future in enumerate(futures):
                            if future.done() and not hasattr(future, "_processed"):
                                try:
                                    result = future.result(timeout=1)
                                    if result:
                                        key = f"{result['model_name']}_{result['timestamp']}"
                                        self.results[key] = result
                                        self._save_to_cache(
                                            self._get_cache_key(
                                                result["model_name"],
                                                result.get("system_prompt"),
                                            ),
                                            result,
                                        )
                                    setattr(future, "_processed", True)
                                    completed += 1
                                    pbar.update(1)
                                except TimeoutError:
                                    print(
                                        "\nWarning: Evaluation timed out for one of the models"
                                    )
                                except Exception as e:
                                    print(f"\nError during evaluation: {str(e)}")
                                    setattr(future, "_processed", True)
                                    completed += 1
                                    pbar.update(1)

                        time.sleep(0.1)

                    pbar.close()

            finally:
                signal.signal(signal.SIGINT, original_sigint)
                self._save_results()
        else:
            print("\nNo new models to evaluate, using cached results")

        missing_models = config_models - set(
            result["model_name"] for result in self.results.values() 
            if result.get("dataset") == "RussianMath" or result.get("dataset") is None
        )
        if missing_models:
            print(f"\nWarning: Missing results for models: {', '.join(missing_models)}")

        self._save_results()

    def evaluate_physics_model(
        self, model_name: str, system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Оценивает одну модель на датасете RussianPhysics.

        Args:
            model_name: Название модели для оценки
            system_prompt: Системный промпт для модели (опционально)

        Returns:
            Словарь с результатами оценки модели
        """
        cache_key = f"{self._get_cache_key(model_name, system_prompt)}_physics"
        cached_result = self._get_cached_result(cache_key)

        if cached_result is not None:
            if self.config.get("debug"):
                print(f"\nUsing cached physics result for {model_name}")
            return cached_result

        if self.config.get("debug"):
            print(f"\nEvaluating {model_name} on RussianPhysics")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model_name = model_name.replace("/", "_")

        temp_config = self.config.copy()
        temp_config["model_list"] = [model_name]
        if system_prompt is not None:
            temp_config[model_name]["system_prompt"] = system_prompt

        temp_config_path = (
            self.output_dir / f"temp_config_physics_{safe_model_name}.yaml"
        )
        with open(temp_config_path, "w") as f:
            yaml.dump(temp_config, f)

        try:
            sampler = OaiSampler(str(temp_config_path))
            evaluator = RussianPhysicsEval(
                equality_checker=self.equality_checker,
                num_examples=self.config.get("num_examples", None),
                debug=self.config.get("debug", False),
            )

            start_time = time.time()
            results = evaluator(sampler)
            evaluation_time = time.time() - start_time

            self._save_detailed_results(
                model_name, results.results, timestamp, "RussianPhysics"
            )

            total_tokens = sum(
                r.tokens for r in results.results if hasattr(r, "tokens")
            )

            model_result = {
                "model_name": model_name,
                "score": results.score,
                "total_tokens": total_tokens,
                "evaluation_time": evaluation_time,
                "system_prompt": system_prompt,
                "timestamp": timestamp,
                "cache_key": cache_key,
                "dataset": "RussianPhysics",
            }

            self._save_to_cache(cache_key, model_result)
            self.results[f"{model_name}_{timestamp}_physics"] = model_result
            self._save_results()

            return model_result

        finally:
            temp_config_path.unlink(missing_ok=True)

    def evaluate_physics_model_parallel(
        self, args: Tuple[str, Optional[str]]
    ) -> Dict[str, Any]:
        """
        Оценивает одну модель на датасете RussianPhysics (для использования в ThreadPoolExecutor).

        Args:
            args: Кортеж (model_name, system_prompt)

        Returns:
            Словарь с результатами оценки модели
        """
        model_name, system_prompt = args
        return self.evaluate_physics_model(model_name, system_prompt)

    def evaluate_physics_models(
        self, system_prompts: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Оценивает все модели на датасете RussianPhysics из конфига параллельно с использованием кэша.

        Args:
            system_prompts: Словарь системных промптов для моделей
                (model_name -> system_prompt)
        """
        if system_prompts is None:
            system_prompts = {}

        measured_models = set()
        for key, result in self.results.items():
            if result.get("dataset") == "RussianPhysics":
                measured_models.add(result.get("model_name"))

        config_models = set(self.config["model_list"])
        new_models = config_models - measured_models
        
        # Определяем модели, которые нужно перетестировать
        models_with_incomplete_results = set()
        if self.retry_incomplete:
            # Получаем модели, у которых есть результаты для RussianPhysics, 
            # но возможно были ошибки или прерывания
            models_with_physics_results = set()
            for key, result in self.results.items():
                if result.get("dataset") == "RussianPhysics" and result["model_name"] in config_models:
                    if result.get("score") is not None and result.get("score") > 0:
                        models_with_physics_results.add(result["model_name"])
            
            models_with_incomplete_results = config_models - models_with_physics_results - new_models
            if models_with_incomplete_results:
                print(f"\nFound models with incomplete RussianPhysics results that will be retested: {', '.join(models_with_incomplete_results)}")

        if new_models:
            print(
                f"\nFound new models to evaluate on RussianPhysics: {', '.join(new_models)}"
            )

        # Объединяем новые модели и модели с неполными результатами
        models_to_evaluate = new_models | models_with_incomplete_results

        if models_to_evaluate:
            uncached_args = [
                (model_name, system_prompts.get(model_name))
                for model_name in models_to_evaluate
            ]

            print(f"\nEvaluating {len(uncached_args)} models on RussianPhysics...")

            def handle_sigint(signum: int, frame: Any) -> None:
                print(
                    "\nGracefully shutting down... Please wait for current evaluations to complete."
                )
                executor.shutdown(wait=True)
                sys.exit(0)

            original_sigint = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, handle_sigint)

            try:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = [
                        executor.submit(self.evaluate_physics_model_parallel, args)
                        for args in uncached_args
                    ]

                    pbar = tqdm(
                        total=len(futures),
                        desc="Evaluating models on RussianPhysics",
                        leave=True,
                    )

                    completed = 0
                    while completed < len(futures):
                        for i, future in enumerate(futures):
                            if future.done() and not hasattr(future, "_processed"):
                                try:
                                    result = future.result(timeout=1)
                                    if result:
                                        key = f"{result['model_name']}_{result['timestamp']}_physics"
                                        self.results[key] = result
                                        self._save_to_cache(
                                            f"{self._get_cache_key(result['model_name'], result.get('system_prompt'))}_physics",
                                            result,
                                        )
                                    setattr(future, "_processed", True)
                                    completed += 1
                                    pbar.update(1)
                                except TimeoutError:
                                    print(
                                        "\nWarning: Evaluation timed out for one of the models"
                                    )
                                except Exception as e:
                                    print(f"\nError during evaluation: {str(e)}")
                                    setattr(future, "_processed", True)
                                    completed += 1
                                    pbar.update(1)

                        time.sleep(0.1)

                    pbar.close()

            finally:
                signal.signal(signal.SIGINT, original_sigint)
                self._save_results()
        else:
            print("\nNo new models to evaluate on RussianPhysics, using cached results")

        missing_models = config_models - set(
            result["model_name"]
            for result in self.results.values()
            if result.get("dataset") == "RussianPhysics"
        )
        if missing_models:
            print(
                f"\nWarning: Missing RussianPhysics results for models: {', '.join(missing_models)}"
            )

        self._save_results()

    def evaluate_demidovich_model(
        self, model_name: str, system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Оценивает одну модель на датасете MathDemon_Demidovich.

        Args:
            model_name: Название модели для оценки
            system_prompt: Системный промпт для модели (опционально)

        Returns:
            Словарь с результатами оценки модели
        """
        cache_key = f"{self._get_cache_key(model_name, system_prompt)}_demidovich"
        cached_result = self._get_cached_result(cache_key)

        if cached_result is not None:
            if self.config.get("debug"):
                print(f"\nUsing cached demidovich result for {model_name}")
            return cached_result

        if self.config.get("debug"):
            print(f"\nEvaluating {model_name} on MathDemonDemidovich")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model_name = model_name.replace("/", "_")

        temp_config = self.config.copy()
        temp_config["model_list"] = [model_name]
        if system_prompt is not None:
            temp_config[model_name]["system_prompt"] = system_prompt

        temp_config_path = (
            self.output_dir / f"temp_config_demidovich_{safe_model_name}.yaml"
        )
        with open(temp_config_path, "w") as f:
            yaml.dump(temp_config, f)

        try:
            sampler = OaiSampler(str(temp_config_path))
            evaluator = MathDemonEval(
                equality_checker=self.equality_checker,
                num_examples=self.config.get("num_examples", None),
                debug=self.config.get("debug", False),
            )

            start_time = time.time()
            results = evaluator(sampler)
            evaluation_time = time.time() - start_time
            self._save_detailed_results(
                model_name, results.results, timestamp, "MathDemon_Demidovich"
            )

            total_tokens = sum(
                r.tokens for r in results.results if hasattr(r, "tokens")
            )

            model_result = {
                "model_name": model_name,
                "score": results.score,
                "total_tokens": total_tokens,
                "evaluation_time": evaluation_time,
                "system_prompt": system_prompt,
                "timestamp": timestamp,
                "cache_key": cache_key,
                "dataset": "MathDemon_Demidovich",
            }

            self._save_to_cache(cache_key, model_result)
            self.results[f"{model_name}_{timestamp}_demidovich"] = model_result
            self._save_results()

            return model_result

        finally:
            temp_config_path.unlink(missing_ok=True)

    def evaluate_demidovich_model_parallel(
        self, args: Tuple[str, Optional[str]]
    ) -> Dict[str, Any]:
        """
        Оценивает одну модель на датасете MathDemon_Demidovich (для использования в ThreadPoolExecutor).

        Args:
            args: Кортеж (model_name, system_prompt)

        Returns:
            Словарь с результатами оценки модели
        """
        model_name, system_prompt = args
        return self.evaluate_demidovich_model(model_name, system_prompt)

    def evaluate_demidovich_models(
        self, system_prompts: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Оценивает все модели на датасете MathDemon_Demidovich из конфига параллельно с использованием кэша.

        Args:
            system_prompts: Словарь системных промптов для моделей
                (model_name -> system_prompt)
        """
        if system_prompts is None:
            system_prompts = {}

        measured_models = set()
        for key, result in self.results.items():
            if result.get("dataset") == "MathDemon_Demidovich":
                measured_models.add(result.get("model_name"))

        config_models = set(self.config["model_list"])
        new_models = config_models - measured_models
        
        # Определяем модели, которые нужно перетестировать
        models_with_incomplete_results = set()
        if self.retry_incomplete:
            # Получаем модели, у которых есть результаты для MathDemon_Demidovich, 
            # но возможно были ошибки или прерывания
            models_with_physics_results = set()
            for key, result in self.results.items():
                if result.get("dataset") == "MathDemon_Demidovich" and result["model_name"] in config_models:
                    if result.get("score") is not None and result.get("score") > 0:
                        models_with_physics_results.add(result["model_name"])
            
            models_with_incomplete_results = config_models - models_with_physics_results - new_models
            if models_with_incomplete_results:
                print(f"\nFound models with incomplete MathDemon_Demidovich results that will be retested: {', '.join(models_with_incomplete_results)}")

        if new_models:
            print(
                f"\nFound new models to evaluate on MathDemon_Demidovich: {', '.join(new_models)}"
            )

        # Объединяем новые модели и модели с неполными результатами
        models_to_evaluate = new_models | models_with_incomplete_results

        if models_to_evaluate:
            uncached_args = [
                (model_name, system_prompts.get(model_name))
                for model_name in models_to_evaluate
            ]

            print(f"\nEvaluating {len(uncached_args)} models on MathDemon_Demidovich...")

            def handle_sigint(signum: int, frame: Any) -> None:
                print(
                    "\nGracefully shutting down... Please wait for current evaluations to complete."
                )
                executor.shutdown(wait=True)
                sys.exit(0)

            original_sigint = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, handle_sigint)

            try:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = [
                        executor.submit(self.evaluate_demidovich_model_parallel, args)
                        for args in uncached_args
                    ]

                    pbar = tqdm(
                        total=len(futures),
                        desc="Evaluating models on MathDemon_Demidovich",
                        leave=True,
                    )

                    completed = 0
                    while completed < len(futures):
                        for i, future in enumerate(futures):
                            if future.done() and not hasattr(future, "_processed"):
                                try:
                                    result = future.result(timeout=1)
                                    if result:
                                        key = f"{result['model_name']}_{result['timestamp']}_demidovich"
                                        self.results[key] = result
                                        self._save_to_cache(
                                            f"{self._get_cache_key(result['model_name'], result.get('system_prompt'))}_demidovich",
                                            result,
                                        )
                                    setattr(future, "_processed", True)
                                    completed += 1
                                    pbar.update(1)
                                except TimeoutError:
                                    print(
                                        "\nWarning: Evaluation timed out for one of the models"
                                    )
                                except Exception as e:
                                    print(f"\nError during evaluation: {str(e)}")
                                    setattr(future, "_processed", True)
                                    completed += 1
                                    pbar.update(1)

                        time.sleep(0.1)

                    pbar.close()

            finally:
                signal.signal(signal.SIGINT, original_sigint)
                self._save_results()
        else:
            print("\nNo new models to evaluate on MathDemon_Demidovich, using cached results")

        missing_models = config_models - set(
            result["model_name"]
            for result in self.results.values()
            if result.get("dataset") == "MathDemon_Demidovich"
        )
        if missing_models:
            print(
                f"\nWarning: Missing MathDemon_Demidovich results for models: {', '.join(missing_models)}"
            )

        self._save_results()


    def _evaluate_subset_parallel(self, subset_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Оценивает все модели на одном подмножестве MathDemon параллельно.

        Args:
            subset_name: Название подмножества

        Returns:
            Словарь результатов по всем моделям для данного подмножества
        """

        def evaluate_model_on_subset(model_name: str) -> Optional[Dict[str, Any]]:
            """
            Оценивает одну модель на одном подмножестве MathDemon.

            Args:
                model_name: Название модели

            Returns:
                Словарь с результатами оценки модели или None при ошибке
            """
            try:
                if self.config.get("debug"):
                    print(
                        f"Starting evaluation of model {model_name} on subset {subset_name}"
                    )

                system_prompt = self.config.get(model_name, {}).get("system_prompt")
                cache_key = (
                    f"{self._get_cache_key(model_name, system_prompt)}_{subset_name}"
                )

                cached_result = self._get_cached_result(cache_key)
                if cached_result:
                    if self.config.get("debug"):
                        print(f"Using cached result for {model_name} on {subset_name}")
                    return cached_result

                evaluator = MathDemonEval(
                    subset_name=subset_name,
                    num_examples=self.config.get("num_examples", None),
                    debug=self.config.get("debug", False),
                )

                evaluator.set_equality_checker(self.equality_checker)

                temp_config = self.config.copy()
                temp_config["model_list"] = [model_name]

                temp_config_path = (
                    self.output_dir
                    / f"temp_config_mathdemon_{model_name}_{subset_name}.yaml"
                )
                with open(temp_config_path, "w", encoding="utf-8") as f:
                    yaml.dump(temp_config, f)

                sampler = OaiSampler(str(temp_config_path))

                start_time = time.time()
                results = evaluator(sampler)
                evaluation_time = time.time() - start_time

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                self._save_detailed_results(
                    model_name, results.results, timestamp, f"MathDemon_{subset_name}"
                )

                total_tokens = sum(
                    r.tokens for r in results.results if hasattr(r, "tokens")
                )

                result_entry = {
                    "model_name": model_name,
                    "score": results.score,
                    "total_tokens": total_tokens,
                    "evaluation_time": evaluation_time,
                    "system_prompt": sampler.system_prompt,
                    "timestamp": timestamp,
                    "dataset": "MathDemon_Dемидович",
                    "subset": subset_name,
                    "cache_key": cache_key,
                }

                self.results[f"{model_name}_MathDemon_{subset_name}_{timestamp}"] = (
                    result_entry
                )

                self._save_to_cache(cache_key, result_entry)

                if not self.config.get("debug"):
                    print(
                        f"Model {model_name} on subset {subset_name}: {results.score:.3f}"
                    )
                else:
                    print(
                        f"Model {model_name} on subset {subset_name} score: {results.score:.3f}, tokens: {total_tokens}, time: {evaluation_time:.1f}s"
                    )

                temp_config_path.unlink(missing_ok=True)

                return result_entry

            except Exception as e:
                print(
                    f"Error evaluating subset {subset_name} for model {model_name}: {str(e)}"
                )
                if "temp_config_path" in locals():
                    temp_config_path.unlink(missing_ok=True)
                return None

        measured_models = set()
        for key, result in self.results.items():
            if (
                result.get("dataset") == "MathDemon_Dемидович"
                and result.get("subset") == subset_name
                and result.get("model_name") in self.config["model_list"]
            ):
                measured_models.add(result.get("model_name"))

        models_to_evaluate = set(self.config["model_list"]) - measured_models

        if not models_to_evaluate:
            print(
                f"All models already evaluated for subset {subset_name}, using cached results"
            )

            results = {}
            for model_name in self.config["model_list"]:
                for key, result in self.results.items():
                    if (
                        result.get("model_name") == model_name
                        and result.get("dataset") == "MathDemon_Dемидович"
                        and result.get("subset") == subset_name
                    ):
                        results[model_name] = result
                        break

            return results

        print(f"Evaluating {len(models_to_evaluate)} models on subset {subset_name}")

        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_model = {
                executor.submit(evaluate_model_on_subset, model_name): model_name
                for model_name in models_to_evaluate
            }

            with tqdm(
                total=len(future_to_model),
                desc=f"Evaluating models on {subset_name}",
                leave=True,
            ) as pbar:
                completed = 0
                while completed < len(future_to_model):
                    for future, model_name in list(future_to_model.items()):
                        if future.done() and not hasattr(future, "_processed"):
                            try:
                                result = future.result(timeout=1)
                                if result:
                                    results[model_name] = result
                                setattr(future, "_processed", True)
                                completed += 1
                                pbar.update(1)
                            except TimeoutError:
                                print(
                                    f"\nWarning: Evaluation timed out for model {model_name}"
                                )
                            except Exception as e:
                                print(
                                    f"\nError during evaluation of model {model_name}: {str(e)}"
                                )
                                setattr(future, "_processed", True)
                                completed += 1
                                pbar.update(1)

                    time.sleep(0.1)

        for model_name in measured_models:
            for key, result in self.results.items():
                if (
                    result.get("model_name") == model_name
                    and result.get("dataset") == "MathDemon_Dемидович"
                    and result.get("subset") == subset_name
                ):
                    results[model_name] = result
                    break

        return results

    def evaluate_math_demon_subsets(self) -> None:
        """
        Оценивает все подсеты из MathDemon_Dемидович для всех моделей из конфига параллельно.
        """
        subsets = [
            "Approximation_by_Polynomials",
            "Continuous_Functions",
            "Convex_Functions",
            "Diﬀerentiation",
            "Improper_Integrals",
            "Infinite_Series",
            "Integration",
            "Sequences_and_Limits",
            "Series_of_Functions",
        ]

        print(f"\nEvaluating MathDemon_Dемидович subsets ({len(subsets)} subsets)")

        all_results = {model: {} for model in self.config["model_list"]}

        for subset in subsets:
            print(f"\nEvaluating subset: {subset} for all models")

            subset_results = self._evaluate_subset_parallel(subset)

            for model_name, result in subset_results.items():
                if result:
                    all_results[model_name][subset] = result

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for model_name, subset_results in all_results.items():
            if subset_results:
                scores = [result["score"] for result in subset_results.values()]
                avg_score = sum(scores) / len(scores)

                total_tokens = sum(
                    result["total_tokens"] for result in subset_results.values()
                )
                total_time = sum(
                    result["evaluation_time"] for result in subset_results.values()
                )

                self.results[f"{model_name}_MathDemon_AllSubsets_{timestamp}"] = {
                    "model_name": model_name,
                    "score": avg_score,
                    "total_tokens": total_tokens,
                    "evaluation_time": total_time,
                    "system_prompt": self.config.get(model_name, {}).get(
                        "system_prompt"
                    ),
                    "timestamp": timestamp,
                    "dataset": "MathDemon_Dемидович",
                    "subset": "AllSubsets",
                }

                print(
                    f"Model {model_name} average score across all MathDemon subsets: {avg_score:.3f}"
                )

        self._save_results()

    def calculate_combined_scores(self) -> None:
        """
        Вычисляет комбинированный скор для каждой модели как полусумму результатов по обоим датасетам.
        """
        print("\nCalculating combined scores for models...")

        model_results = {}

        for key, result in self.results.items():
            model_name = result["model_name"]

            if key.endswith("_Combined"):
                continue

            if model_name not in model_results:
                model_results[model_name] = {
                    "RussianMath": None,
                    "RussianPhysics": None,
                    "MathDemon_Demidovich": None
                }

            dataset = result.get("dataset", "RussianMath")

            if dataset == "RussianMath" and (
                model_results[model_name]["RussianMath"] is None
                or result["score"] > model_results[model_name]["RussianMath"]["score"]
            ):
                model_results[model_name]["RussianMath"] = result

            elif dataset == "RussianPhysics" and (
                model_results[model_name]["RussianPhysics"] is None
                or result["score"]
                > model_results[model_name]["RussianPhysics"]["score"]
            ):
                model_results[model_name]["RussianPhysics"] = result

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for model_name, results in model_results.items():
            if results["RussianMath"] and results["RussianPhysics"]:
                math_score = results["RussianMath"]["score"]
                physics_score = results["RussianPhysics"]["score"]
                demidovich_score = results["MathDemon_Demidovich"]["score"]
                combined_score = (math_score + physics_score + demidovich_score) / 3.0

                total_tokens = results["RussianMath"].get("total_tokens", 0) + results[
                    "RussianPhysics"
                ].get("total_tokens", 0) + results["MathDemon_Demidovich"].get("total_tokens", 0)
                total_time = results["RussianMath"].get("evaluation_time", 0) + results[
                    "RussianPhysics"
                ].get("evaluation_time", 0) + results["MathDemon_Demidovich"].get("evaluation_time", 0)

                system_prompt = results["RussianMath"].get("system_prompt") or results[
                    "RussianPhysics"
                ].get("system_prompt") or results["MathDemon_Demidovich"].get("system_prompt")

                self.results[f"{model_name}_Combined_{timestamp}"] = {
                    "model_name": model_name,
                    "score": combined_score,
                    "math_score": math_score,
                    "physics_score": physics_score,
                    "demidovich_score": demidovich_score,
                    "total_tokens": total_tokens,
                    "evaluation_time": total_time,
                    "system_prompt": system_prompt,
                    "timestamp": timestamp,
                    "dataset": "Combined",
                }

                print(
                    f"Model {model_name} combined score: {combined_score:.3f} (Math: {math_score:.3f}, Physics: {physics_score:.3f}, Demidovich: {demidovich_score:.3f})"
                )

            elif results["RussianMath"]:
                print(
                    f"Warning: Model {model_name} has only RussianMath results, skipping combined score calculation"
                )

            elif results["RussianPhysics"]:
                print(
                    f"Warning: Model {model_name} has only RussianPhysics results, skipping combined score calculation"
                )
            elif results["MathDemon_Demidovich"]:
                print(
                    f"Warning: Model {model_name} has only MathDemon_Demidovich results, skipping combined score calculation"
                )

        self._save_results()

    # МЕТОДЫ ГЕНЕРАЦИИ РЕЗУЛЬТАТОВ

    def generate_markdown(self) -> str:
        """
        Генерирует markdown с результатами оценок моделей.

        Returns:
            Строка с markdown-разметкой таблицы лидерборда
        """
        md = "# Math Evaluation Leaderboard\n\n"
        md += f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        md += "| Model | Combined Score | RussianMath Score | RussianPhysics Score | Demidovich Score | Tokens Used | System Prompt | Evaluation Time | Details |\n"
        md += "|-------|---------------|-------------------|----------------------|------------------|-------------|---------------|----------------|--------|\n"

        self._prepare_combined_reports()

        model_data = {}

        for key, result in self.results.items():
            model_name = result["model_name"]

            if model_name not in model_data:
                model_data[model_name] = {
                    "combined": None,
                    "russianmath": None,
                    "physics": None,
                    "demidovich": None
                }

            dataset = result.get("dataset")

            if dataset == "Combined":
                model_data[model_name]["combined"] = result
            elif dataset == "RussianMath" or (dataset is None):
                if (
                    not model_data[model_name]["russianmath"]
                    or result["score"] > model_data[model_name]["russianmath"]["score"]
                ):
                    model_data[model_name]["russianmath"] = result
            elif dataset == "RussianPhysics":
                if (
                    not model_data[model_name]["physics"]
                    or result["score"] > model_data[model_name]["physics"]["score"]
                ):
                    model_data[model_name]["physics"] = result
            elif dataset == "MathDemon_Demidovich":
                if (
                    not model_data[model_name]["demidovich"]
                    or result["score"] > model_data[model_name]["demidovich"]["score"]
                ):
                    model_data[model_name]["demidovich"] = result

        def get_sort_score(model_name: str) -> float:
            data = model_data[model_name]
            if data["combined"]:
                return data["combined"]["score"]
            elif data["russianmath"]:
                return data["russianmath"]["score"]
            return 0

        sorted_models = sorted(model_data.keys(), key=get_sort_score, reverse=True)

        for model_name in sorted_models:
            data = model_data[model_name]

            if not data["russianmath"] and not data["physics"]:
                continue

            combined_score = data["combined"]["score"] if data["combined"] else "-"
            rm_score = data["russianmath"]["score"] if data["russianmath"] else "-"
            physics_score = data["physics"]["score"] if data["physics"] else "-"
            demidovich_score = data["demidovich"]["score"] if data["demidovich"] else "-"

            total_tokens = 0
            if data["russianmath"]:
                total_tokens += data["russianmath"].get("total_tokens", 0)
            if data["physics"]:
                total_tokens += data["physics"].get("total_tokens", 0)
            if data["demidovich"]:
                total_tokens += data["demidovich"].get("total_tokens", 0)

            system_prompt = None
            for result_type in ["russianmath", "physics", "demidovich", "combined"]:
                if data[result_type] and data[result_type].get("system_prompt"):
                    system_prompt = data[result_type]["system_prompt"]
                    break

            if system_prompt and len(system_prompt) > 30:
                system_prompt = system_prompt[:27] + "..."
            elif not system_prompt:
                system_prompt = "None"

            eval_time = 0
            if data["russianmath"]:
                eval_time += data["russianmath"].get("evaluation_time", 0)
            if data["physics"]:
                eval_time += data["physics"].get("evaluation_time", 0)
            if data["demidovich"]:
                eval_time += data["demidovich"].get("evaluation_time", 0)

            details = ""
            safe_model_name = model_name.replace("/", "_")

            if data["russianmath"] and data["physics"] and data["demidovich"]:
                math_timestamp = data["russianmath"]["timestamp"]
                physics_timestamp = data["physics"]["timestamp"]
                demidovich_timestamp = data["demidovich"]["timestamp"]

                combined_report = (
                    self.details_dir
                    / safe_model_name
                    / f"details_{math_timestamp}_combined.md"
                )
                if combined_report.exists():
                    details = f"[Combined](results/details/{safe_model_name}/details_{math_timestamp}_combined.md)"
                else:
                    combined_report = (
                        self.details_dir
                        / safe_model_name
                        / f"details_{physics_timestamp}_combined.md"
                    )
                    if combined_report.exists():
                        details = f"[Combined](results/details/{safe_model_name}/details_{physics_timestamp}_combined.md)"
                    else:
                        details = f"[Math](results/details/{safe_model_name}/details_{math_timestamp}.md) [Physics](results/details/{safe_model_name}/details_{physics_timestamp}_RussianPhysics.md)"
            else:
                if data["russianmath"]:
                    timestamp = data["russianmath"]["timestamp"]
                    details = f"[Math](results/details/{safe_model_name}/details_{timestamp}.md)"
                elif data["physics"]:
                    timestamp = data["physics"]["timestamp"]
                    details = f"[Physics](results/details/{safe_model_name}/details_{timestamp}_RussianPhysics.md)"

            md += f"| {model_name} "
            md += f"| {combined_score if isinstance(combined_score, str) else f'{combined_score:.3f}'} "
            md += f"| {rm_score if isinstance(rm_score, str) else f'{rm_score:.3f}'} "
            md += f"| {physics_score if isinstance(physics_score, str) else f'{physics_score:.3f}'} "
            md += f"| {demidovich_score if isinstance(demidovich_score, str) else f'{demidovich_score:.3f}'} "
            md += f"| {total_tokens} "
            md += f"| {system_prompt} "
            md += f"| {eval_time:.1f}s "
            md += f"| {details} |\n"

        with open(self.output_dir / "leaderboard.md", "w", encoding="utf-8") as f:
            f.write(md)

        return md


def main() -> None:
    """
    Пример использования класса Leaderboard.
    """
    leaderboard = Leaderboard("configs/run.yaml")

    system_prompts = {
        "gpt-4-1106-preview": "You are a helpful math assistant. Answer in Russian.",
        "gpt-3.5-turbo-0125": "Solve math problems step by step. Answer in Russian.",
        "gpt-4o-mini": "You are a math expert. Provide detailed solutions in Russian.",
    }

    leaderboard.evaluate_all_models(system_prompts)

    md = leaderboard.generate_markdown()
    print("Leaderboard generated!")
    print(md)


if __name__ == "__main__":
    main()
