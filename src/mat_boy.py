import re
from typing import Dict, List, Optional
from datasets import load_dataset
from .common import check_equality
from .types import Eval, EvalResult, SamplerBase, SingleEvalResult
from . import common

QUERY_TEMPLATE_RU = """
Реши следующую математическую задачу пошагово. Последняя строка твоего ответа должна быть в формате Ответ: $ANSWER (без кавычек, скобок и текстового форматирования), где $ANSWER - это ответ на задачу. Ответ должен быть точным, если необходимо - несократимой дробью через точку. Если задача подразумевает перечисления - выпиши все ответы слитно без разделителей. Если в задаче требуется найти несколько неизвестных - перечисляй их через ";". Используй те единицы измерения, которые содержатся в условии, если в нем не сказано обратного, сами единицы измерения в ответ не записывай. Если требуется выбрать что-то перечислительное - напиши только само число. После ответа не пиши ничего. Далее сама задача:

{task}

Не забудь написать ответ в отдельной строке после "Ответ:", без использования команды \\boxed и в нужном формате.
""".strip()

PHYSICS_TEMPLATE_RU = """
Реши следующую задачу по физике пошагово. Последняя строка твоего ответа должна быть в формате Ответ: $ANSWER (без кавычек, скобок и текстового форматирования), где $ANSWER - это ответ на задачу. Ответ должен быть точным, если необходимо - несократимой дробью через точку. Если задача подразумевает перечисления - выпиши все ответы слитно без разделителей. Если в задаче требуется найти несколько неизвестных - перечисляй их через ";". Используй те единицы измерения, которые содержатся в условии, если в нем не сказано обратного, сами единицы измерения в ответ не записывай. После ответа не пиши ничего. Если требуется выбрать что-то перечислительное - напиши только само число. После ответа не пиши ничего. Далее сама задача:

{task}

Не забудь написать ответ в отдельной строке после "Ответ:", без использования команды \\boxed и в нужном формате.
""".strip()


class RussianMathEval(Eval):
    """
    Класс для оценки языковых моделей на русскоязычных математических задачах.
    """

    def __init__(
        self,
        equality_checker: SamplerBase,
        num_examples: Optional[int] = None,
        n_repeats: int = 1,
        debug: bool = False,
    ) -> None:
        """
        Инициализирует оценку на русскоязычных математических задачах.

        Args:
            equality_checker: Объект для проверки равенства ответов
            num_examples: Количество примеров для оценки (по умолчанию 5)
            n_repeats: Количество повторений набора примеров
            debug: Режим отладки для подробного вывода
        """
        dataset = load_dataset("Vikhrmodels/russian_math")
        examples = [
            {"task": row["task"], "Answer": row["short answer"]}
            for row in dataset["train"]
        ]

        if num_examples and num_examples > 0:
            examples = examples[:num_examples]

        self.examples: List[Dict[str, str]] = examples * n_repeats
        self.equality_checker: SamplerBase = equality_checker
        self.debug: bool = debug

        if self.debug:
            print(f"Loaded {len(self.examples)} examples for evaluation")

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        """
        Выполняет оценку модели на математических задачах.

        Args:
            sampler: Модель для оценки

        Returns:
            Результат оценки модели
        """

        def fn(row: Dict[str, str]) -> SingleEvalResult:
            """
            Обрабатывает один пример задачи.

            Args:
                row: Словарь с задачей и ответом

            Returns:
                Результат оценки для одного примера
            """
            if self.debug:
                print("\nDebug: Processing example")
                print(f"Task: {row['task']}")
                print(f"Expected answer: {row['Answer']}")

            prompt_messages = [
                sampler._pack_message(
                    content=QUERY_TEMPLATE_RU.format(**row), role="user"
                )
            ]

            response_text, metadata = sampler(prompt_messages, return_metadata=True)

            answer_pattern = r"(?:Answer|Ответ):\s*(.+)$"
            matches = list(re.finditer(answer_pattern, response_text, re.MULTILINE))
            extracted_answer = matches[-1].group(1).strip() if matches else None

            if self.debug:
                print(f"Extracted answer: {extracted_answer}")

            score = float(
                check_equality(
                    self.equality_checker, str(row["Answer"]), extracted_answer
                )
            )

            if self.debug:
                print(f"Score: {score}")
                print(f"Tokens used: {metadata.get('total_tokens', 0)}")

            html = common.jinja_env.from_string(common.HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["Answer"],
                extracted_answer=extracted_answer,
                tokens=metadata.get("total_tokens", 0),
            )

            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(
                html=html,
                score=score,
                convo=convo,
                correct_answer=row["Answer"],
                extracted_answer=extracted_answer,
                tokens=metadata.get("total_tokens", 0),
            )

        results = common.map_with_progress(
            fn, self.examples, model_name=sampler.model_name
        )
        return common.aggregate_results(results)


class RussianPhysicsEval(Eval):
    """
    Класс для оценки языковых моделей на русскоязычных задачах по физике.
    """

    def __init__(
        self,
        equality_checker: SamplerBase,
        num_examples: Optional[int] = None,
        n_repeats: int = 1,
        debug: bool = False,
    ) -> None:
        """
        Инициализирует оценку на русскоязычных задачах по физике.

        Args:
            equality_checker: Объект для проверки равенства ответов
            num_examples: Количество примеров для оценки (по умолчанию 5)
            n_repeats: Количество повторений набора примеров
            debug: Режим отладки для подробного вывода
        """
        dataset = load_dataset("Vikhrmodels/russian_physics")
        examples = [
            {"task": row["task"], "Answer": row["answer"]} for row in dataset["train"]
        ]

        if num_examples and num_examples > 0:
            examples = examples[:num_examples]

        self.examples: List[Dict[str, str]] = examples * n_repeats
        self.equality_checker: SamplerBase = equality_checker
        self.debug: bool = debug

        if self.debug:
            print(f"Loaded {len(self.examples)} physics examples for evaluation")

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        """
        Выполняет оценку модели на задачах по физике.

        Args:
            sampler: Модель для оценки

        Returns:
            Результат оценки модели
        """

        def fn(row: Dict[str, str]) -> SingleEvalResult:
            """
            Обрабатывает один пример задачи.

            Args:
                row: Словарь с задачей и ответом

            Returns:
                Результат оценки для одного примера
            """
            if self.debug:
                print("\nDebug: Processing physics example")
                print(f"Task: {row['task']}")
                print(f"Expected answer: {row['Answer']}")

            prompt_messages = [
                sampler._pack_message(
                    content=PHYSICS_TEMPLATE_RU.format(**row), role="user"
                )
            ]

            response_text, metadata = sampler(prompt_messages, return_metadata=True)

            answer_pattern = r"(?:Answer|Ответ):\s*(.+)$"
            matches = list(re.finditer(answer_pattern, response_text, re.MULTILINE))
            extracted_answer = matches[-1].group(1).strip() if matches else None

            if self.debug:
                print(f"Extracted answer: {extracted_answer}")

            score = float(
                check_equality(
                    self.equality_checker, str(row["Answer"]), extracted_answer
                )
            )

            if self.debug:
                print(f"Score: {score}")
                print(f"Tokens used: {metadata.get('total_tokens', 0)}")

            html = common.jinja_env.from_string(common.HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["Answer"],
                extracted_answer=extracted_answer,
                tokens=metadata.get("total_tokens", 0),
            )

            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(
                html=html,
                score=score,
                convo=convo,
                correct_answer=row["Answer"],
                extracted_answer=extracted_answer,
                tokens=metadata.get("total_tokens", 0),
            )

        results = common.map_with_progress(
            fn, self.examples, model_name=sampler.model_name
        )
        return common.aggregate_results(results)


class MathDemonEval(Eval):
    """
    Класс для оценки языковых моделей на задачах из учебника Демидовича.
    """

    def __init__(
        self,
        equality_checker: SamplerBase,
        num_examples: Optional[int] = None,
        n_repeats: int = 1,
        debug: bool = False,
    ) -> None:
        """
        Инициализирует оценку на подсетах MathDemon_Demidovich.

        Args:
            subset_name: Название подмножества задач (раздел учебника)
            num_examples: Количество примеров для оценки (по умолчанию 1)
            debug: Режим отладки для подробного вывода
        """
        tasks = ['Approximation_by_Polynomials', 'Continuous_Functions', 'Convex_Functions', 'Diﬀerentiation', 'Improper_Integrals', 'Infinite_Series', 'Integration', 'Sequences_and_Limits', 'Series_of_Functions']
        dss = [load_dataset("Vikhrmodels/MathDemon_Demidovich", task) for task in tasks]
        examples = []
        for dataset in dss:
            examples += [
                {"task": row["problem"], "Answer": row["answer"]}
                for row in dataset["train"]
            ]

        if num_examples and num_examples > 0:
            examples = examples[:num_examples]

        self.examples: List[Dict[str, str]] = examples
        self.debug: bool = debug
        self.equality_checker: Optional[SamplerBase] = equality_checker

        if self.debug:
            print(f"Loaded {len(self.examples)} examples")

    def set_equality_checker(self, equality_checker: SamplerBase) -> None:
        """
        Устанавливает объект для проверки равенства ответов.

        Args:
            equality_checker: Объект для проверки равенства ответов
        """
        self.equality_checker = equality_checker

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        """
        Выполняет оценку модели на задачах из Демидовича.

        Args:
            sampler: Модель для оценки

        Returns:
            Результат оценки модели
        """

        def fn(row: Dict[str, str]) -> SingleEvalResult:
            """
            Обрабатывает один пример задачи.

            Args:
                row: Словарь с задачей и ответом

            Returns:
                Результат оценки для одного примера
            """
            if self.debug:
                print("\nDebug: Processing example")
                print(f"Task: {row['task']}")
                print(f"Expected answer: {row['Answer']}")

            prompt_messages = [
                sampler._pack_message(
                    content=QUERY_TEMPLATE_RU.format(task=row["task"]), role="user"
                )
            ]

            response_text, metadata = sampler(prompt_messages, return_metadata=True)

            answer_pattern = r"(?:Answer|Ответ):\s*(.+)$"
            matches = list(re.finditer(answer_pattern, response_text, re.MULTILINE))
            extracted_answer = matches[-1].group(1).strip() if matches else None

            if self.debug:
                print(f"Extracted answer: {extracted_answer}")

            score = 0.0
            if self.equality_checker:
                score = float(
                    check_equality(
                        self.equality_checker, str(row["Answer"]), extracted_answer
                    )
                )

            if self.debug:
                print(f"Score: {score}")
                print(f"Tokens used: {metadata.get('total_tokens', 0)}")

            html = common.jinja_env.from_string(common.HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["Answer"],
                extracted_answer=extracted_answer,
                tokens=metadata.get("total_tokens", 0),
            )

            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(
                html=html,
                score=score,
                convo=convo,
                correct_answer=row["Answer"],
                extracted_answer=extracted_answer,
                tokens=metadata.get("total_tokens", 0),
            )

        results = common.map_with_progress(
            fn, self.examples, model_name=sampler.model_name
        )
        return common.aggregate_results(results)
