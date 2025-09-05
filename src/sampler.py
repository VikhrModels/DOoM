import yaml
from typing import List, Dict, Tuple, Union, Any, Optional
import openai
import time
import re
import threading
import logging
import traceback
import json
from json.decoder import JSONDecodeError
from urllib.parse import urlparse

from gigachat import GigaChat
from gigachat.models import Chat, Messages

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("api_requests.log", mode="a")],
)
logger = logging.getLogger("sampler")

API_MAX_RETRY = 17
API_RETRY_SLEEP = 7
API_ERROR_OUTPUT = "Error during API call. Please try again."

API_ERROR_PATTERNS = [
    r"###\s*Model\s*Response\s*Error\s*during\s*API\s*call",
    r"Error\s*during\s*API\s*call.*try\s*again",
    r"API\s*(call|request)\s*(failed|error|timeout)",
    r"Exception\s*occurred.*API",
    r"(failed|error|unable)\s*to\s*(generate|get|fetch)\s*response",
    r"The\s*model\s*did\s*not\s*provide\s*a\s*(response|answer)",
    r"^(Error:|Warning:|Exception:|API Error:)",
]

JSON_ERROR_MAX_RETRY = 12
JSON_ERROR_RETRY_DELAY = 5


class RateLimiter:
    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.last_request_time = 0
        self.lock = threading.Lock()

    def wait_if_needed(self) -> None:
        if self.delay <= 0:
            return

        with self.lock:
            current_time = time.time()
            elapsed = current_time - self.last_request_time
            if elapsed < self.delay:
                wait_time = self.delay - elapsed
                time.sleep(wait_time)
            self.last_request_time = time.time()


def safe_response_dump(response: Any) -> str:
    if response is None:
        return "None"
    try:
        if isinstance(response, (dict, list, str, int, float, bool)):
            return json.dumps(response, ensure_ascii=False, indent=2, default=str)
        return f"{type(response).__name__}: {str(response)}"
    except Exception as e:
        return f"[Error serializing {type(response).__name__}: {str(e)}]"


class OaiSampler:
    _rate_limiters = {}
    _rate_limiters_lock = threading.Lock()

    @classmethod
    def get_rate_limiter(cls, api_type: str, model_name: str, delay: float) -> RateLimiter:
        """
        Получает или создает rate limiter для конкретной модели.
        
        :param api_type: Тип API (openai, openrouter, gigachat)
        :param model_name: Имя модели
        :param delay: Задержка между запросами в секундах
        :return: Объект RateLimiter для данной модели
        """
        key = f"{api_type}_{model_name}"
        with cls._rate_limiters_lock:
            if key not in cls._rate_limiters:
                cls._rate_limiters[key] = RateLimiter(delay)
            return cls._rate_limiters[key]

    def __init__(self, config_path: str):
        with open(config_path, "r", encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        if not isinstance(self.config, dict):
            raise ValueError("Invalid configuration format")

        model_name = self.config["model_list"][0]
        self.model_config = self.config.get(model_name, {})
        self.api_type = self.model_config.get("api_type", "openai").strip().lower()

        endpoint = {}
        if "endpoints" in self.model_config:
            endpoints = self.model_config["endpoints"]
            if endpoints and isinstance(endpoints, list) and endpoints[0]:
                endpoint = endpoints[0]

        # API credentials setup
        if self.api_type in ["openai", "openrouter"]:
            self.api_key = endpoint.get("api_key") or self.config.get("api_key")
            if not self.api_key:
                raise ValueError(f"API key required for {self.api_type}")

            self.base_url = endpoint.get("api_base") or endpoint.get("base_url")

            if self.api_type == "openrouter":
                if not self.base_url:
                    self.base_url = "https://openrouter.ai/api/v1"
                elif not self.base_url.startswith("https://openrouter.ai"):
                    logger.warning("OpenRouter base_url should start with 'https://openrouter.ai'")

                self.referer_url = endpoint.get("referer_url") or endpoint.get("HTTP-Referer")
                self.app_title = endpoint.get("app_title") or endpoint.get("X-Title")
                
                # Валидация referer_url для OpenRouter
                if not self.referer_url:
                    logger.warning("OpenRouter: referer_url не указан, может возникнуть ошибка 403")
                elif not self._validate_referer_url(self.referer_url):
                    logger.warning(f"OpenRouter: некорректный формат referer_url: {self.referer_url}")
            else:
                self.referer_url = None
                self.app_title = None

            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            if self.api_type == "openrouter" and (self.referer_url or self.app_title):
                headers = {}
                if self.referer_url and isinstance(self.referer_url, str) and self.referer_url.strip():
                    headers["HTTP-Referer"] = self.referer_url.strip()
                if self.app_title and isinstance(self.app_title, str) and self.app_title.strip():
                    headers["X-Title"] = self.app_title.strip()
                if headers:
                    client_kwargs["default_headers"] = headers

            self.client = openai.OpenAI(**client_kwargs)

        elif self.api_type == "gigachat":
            self.credentials = endpoint.get("credentials") or self.config.get("credentials")
            if not self.credentials:
                raise ValueError("Credentials required for GigaChat")
            self.base_url = endpoint.get("base_url", "https://gigachat.devices.sberbank.ru/api/v1")
            self.client = None
        else:
            raise ValueError(f"Unsupported API type: {self.api_type}")

        self.model_name = self.model_config.get("model_name", model_name)
        self.temperature = self.config.get("temperature", 0.0)
        self.max_tokens = self.model_config.get("max_tokens", self.config.get("max_tokens", 2048))
        self.system_prompt = self.model_config.get("system_prompt", None)
        self.debug = self.config.get("debug", False)
        self.request_delay = self.model_config.get("request_delay", self.config.get("request_delay", 0.0))
        self.rate_limiter = self.get_rate_limiter(self.api_type, self.model_name, self.request_delay)

    def _validate_referer_url(self, url: str) -> bool:
        """
        Проверяет корректность referer URL для OpenRouter.
        
        :param url: URL для проверки
        :return: True если URL корректный, False в противном случае
        """
        if not url or not isinstance(url, str):
            return False
        
        try:
            parsed = urlparse(url.strip())
            return all([
                parsed.scheme in ['http', 'https'],
                parsed.netloc,
                not parsed.netloc.startswith('localhost'),
                '.' in parsed.netloc
            ])
        except Exception:
            return False

    def _handle_openrouter_error(self, error: Exception) -> str:
        """
        Обрабатывает специфичные для OpenRouter ошибки.
        
        :param error: Исходное исключение
        :return: Понятное описание ошибки
        """
        error_msg = str(error).lower()
        
        if "referer" in error_msg or "403" in error_msg:
            return (f"OpenRouter ошибка: Некорректный или отсутствующий HTTP-Referer заголовок. "
                   f"Проверьте параметр referer_url в конфигурации.")
        elif "rate limit" in error_msg or "429" in error_msg:
            return (f"OpenRouter: превышен лимит запросов. "
                   f"Увеличьте параметр request_delay в конфигурации.")
        elif "insufficient" in error_msg and "credit" in error_msg:
            return "OpenRouter: недостаточно средств на аккаунте"
        elif "unauthorized" in error_msg or "401" in error_msg:
            return "OpenRouter: некорректный API ключ"
        elif "model" in error_msg and ("not found" in error_msg or "unavailable" in error_msg):
            return f"OpenRouter: модель '{self.model_name}' недоступна или не найдена"
        
        return f"OpenRouter API ошибка: {error}"

    def _pack_message(self, content: str, role: str = "user") -> Dict[str, str]:
        return {"role": role, "content": content}


    def contains_error_patterns(self, text: str) -> bool:
        if not text:
            return True
        for pattern in API_ERROR_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def chat_completion_gigachat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Tuple[str, Dict[str, int]]:
        output = API_ERROR_OUTPUT
        metadata = {"total_tokens": 0}
        logger.info(f"API request: [{model}] (GigaChat)")

        client = GigaChat(
            base_url=self.base_url,
            credentials=self.credentials,
            scope="GIGACHAT_API_CORP",
            verify_ssl_certs=False,
            timeout=60.0
        )

        top_p = 1.0
        if temperature == 0:
            temperature = 1
            top_p = 0

        giga_messages = [Messages(role=msg["role"], content=msg["content"]) for msg in messages]
        chat = Chat(messages=giga_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p)

        for attempt in range(API_MAX_RETRY):
            if attempt > 0:
                retry_delay = API_RETRY_SLEEP * (1 + attempt * 0.5)
                logger.info(f"Model [{model}]: Retry #{attempt + 1}, delay: {retry_delay:.1f}s")
                time.sleep(retry_delay)

            try:
                response = client.chat(chat)
                output = response.choices[0].message.content

                if self.contains_error_patterns(output):
                    error_msg = output.strip()[:100]
                    logger.warning(f"Model [{model}]: Error pattern in response: {error_msg}")
                    continue

                if response.usage:
                    metadata["prompt_tokens"] = response.usage.prompt_tokens
                    metadata["completion_tokens"] = response.usage.completion_tokens
                    metadata["total_tokens"] = response.usage.total_tokens

                logger.info(f"Model [{model}]: Success, tokens: {metadata['total_tokens']}")
                break
            except Exception as e:
                logger.error(f"Model [{model}]: Error: {type(e).__name__}: {str(e)}")
                if attempt == API_MAX_RETRY - 1:
                    output = f"Error during API call: {str(e)}"

        return output, metadata

    def __call__(
        self, messages: List[Dict[str, str]], return_metadata: bool = False
    ) -> Union[str, Tuple[str, Dict[str, int]]]:
        self.rate_limiter.wait_if_needed()

        if self.debug and messages:
            msg_preview = messages[0]["content"][:50] + "..." if len(messages[0]["content"]) > 50 else messages[0]["content"]
            logger.debug(f"Sending request to {self.model_name}: {msg_preview}")

        if self.system_prompt:
            messages = [self._pack_message(self.system_prompt, "system")] + messages

        if self.api_type == "gigachat":
            result, metadata = self.chat_completion_gigachat(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return (result, metadata) if return_metadata else result

        else:  # openai/openrouter
            for json_retry in range(JSON_ERROR_MAX_RETRY):
                try:
                    return self._process_openai_request(messages, return_metadata)
                except Exception as e:
                    is_retryable = (
                        isinstance(e, JSONDecodeError)
                        or "JSONDecodeError" in str(e)
                        or "Expecting value" in str(e)
                        or isinstance(e, TypeError)
                    )
                    if is_retryable and json_retry < JSON_ERROR_MAX_RETRY - 1:
                        retry_delay = JSON_ERROR_RETRY_DELAY * (1 + json_retry * 0.5)
                        time.sleep(retry_delay)
                        continue
                    logger.error(f"API error: {type(e).__name__}: {str(e)}")
                    raise

    def _process_openai_request(
        self, messages: List[Dict[str, str]], return_metadata: bool = False
    ):
        api_args = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        api_label = "OpenRouter" if self.api_type == "openrouter" else "OpenAI"
        logger.info(f"API request: [{self.model_name}] ({api_label})")

        try:
            response = self.client.chat.completions.create(**api_args)
        except Exception as e:
            if self.api_type == "openrouter":
                error_msg = self._handle_openrouter_error(e)
                logger.error(f"OpenRouter ошибка: {error_msg}")
                raise Exception(error_msg) from e
            else:
                logger.error(f"Model error: {type(e).__name__}: {str(e)}")
                raise

        metadata = {"total_tokens": 0}
        if response.usage:
            metadata["prompt_tokens"] = response.usage.prompt_tokens
            metadata["completion_tokens"] = response.usage.completion_tokens
            metadata["total_tokens"] = response.usage.total_tokens

        try:
            content = response.choices[0].message.content
            if self.contains_error_patterns(content):
                error_part = content[:100] + "..." if len(content) > 100 else content
                logger.warning(f"Model error patterns in response: {error_part}")
            else:
                logger.info(f"Model success: tokens={metadata['total_tokens']}")

            return (content, metadata) if return_metadata else content
        except AttributeError:
            error_msg = f"Invalid response structure: {safe_response_dump(response)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
