import asyncio
import random
import json
import re
import threading
import g4f

import json


import asyncio
import random
from langchain.callbacks.manager import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from typing import Callable, List, Any, Optional
from langchain.llms.base import LLM
import g4f

g4f.logging = False  # enable logging


def safe_json_parse(json_str, expected_keys):
    """
    A function to parse JSON strings and ensure they contain the expected keys.

    Args:
    - json_str (str): The JSON string to parse.
    - expected_keys (list[str]): A list of keys that are expected to be in the parsed JSON.

    Returns:
    - dict: Parsed JSON data.

    Raises:
    - ValueError: If parsing fails or expected keys are missing.
    """

    # Step 1: Text Cleaning
    clean_str = json_str.strip()
    clean_str = re.sub(r'[\n\t\r]', '', clean_str)

    # Extract all JSON objects from the string using a more greedy approach
    json_objects = re.findall(r'{.*}', clean_str, re.DOTALL)

    if not json_objects:
        raise ValueError("No JSON objects found in the provided string.")

    # Use the last JSON object
    clean_str = json_objects[-1]

    # Brace Matching
    if clean_str.count('{') != clean_str.count('}'):
        raise ValueError("Mismatched braces in the provided JSON string.")

    # JSON Parsing
    try:
        data = json.loads(clean_str)
    except json.JSONDecodeError as e:
        error_pos = getattr(e, 'pos', None)
        if error_pos:
            context = 20  # number of characters to show on each side of the error position
            start = max(0, error_pos - context)
            end = min(len(clean_str), error_pos + context)
            excerpt = clean_str[start:end]
            error_message = f"Error at position {error_pos}. Surrounding text: ...{excerpt}..."
        else:
            error_message = str(e)
        raise ValueError(f"Error parsing JSON string: {error_message}")

    # Expected Keys Check
    missing_keys = [key for key in expected_keys if key not in data]
    if missing_keys:
        raise ValueError(
            f"Expected keys {missing_keys} missing from the parsed data.")

    return data


def check_prompt_length(prompt: str, max_length: int):
    if len(prompt) > max_length * 0.9:
        print("WARNING: String length has reached 90% of the maximum length.")
    if len(prompt) > max_length:
        raise ValueError(
            "ERROR: String length has exceeded the maximum length.")


MAX_PROMPT_LENGTH = 32000


class G4FLLM(LLM):

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: List[str] | None = None, run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> str:
        check_prompt_length(prompt, max_length=MAX_PROMPT_LENGTH)

        print("WARNING: NOT IMPLEMENTED _CALL: RETRUNING 'TODO'")
        return "TODO"

    async def _acall(self, prompt: str, stop: List[str] | None = None, run_manager: AsyncCallbackManagerForLLMRun | None = None, **kwargs: Any) -> str:
        check_prompt_length(prompt, max_length=MAX_PROMPT_LENGTH)
        #
        for _ in range(10):  # number of retries
            try:
                response = await g4f.ChatCompletion.create_async(
                    model=g4f.models.gpt_4,
                    provider=g4f.Provider.Bing,
                    messages=[{"role": "user", "content": prompt}],
                    auth=True
                )
                return response
            except Exception as e:
                print(f"Error getting response: {e}")
                print(
                    f"PROMPT LENGTH: {len(prompt)} \nMAX LENGTH: {MAX_PROMPT_LENGTH}")
                # adjust sleep duration as needed
                await asyncio.sleep(random.uniform(0.1, 1))
        raise RuntimeError("Failed to get response after multiple retries")

    # ... (rest of your code remains the same)

    async def acall(self, prompt: str, stop: List[str] | None = None, run_manager: AsyncCallbackManagerForLLMRun | None = None, **kwargs: Any) -> str:
        return await self._acall(prompt, stop, run_manager, **kwargs)

    # def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
    #     out = gpt4free.Completion.create(Provider.You, prompt=prompt)
    #     # If stop words are provided, find the earliest occurrence of any stop word in the output
    #     if stop:
    #         stop_indexes = (out.find(s) for s in stop if s in out)
    #         min_stop = min(stop_indexes, default=-1)
    #         # If a stop word is found, truncate the output at that poiniit
    #         if min_stop > -1:
    #             out = out[:min_stop]
    #     return out


class SemaphoreLLMChain:
    def __init__(self, chain, semaphore: asyncio.Semaphore, verbose: bool = False):
        """
        Initializes the SemaphoreLLMChain.
        :param chain: The chain to be executed
        :param semaphore: An asyncio.Semaphore instance to control concurrency
        :param verbose: Flag to enable/disable verbose logging
        """
        self.chain = chain
        self.semaphore = semaphore
        self.verbose = verbose

    def _log(self, message: str):
        """
        Logs a message if verbose is enabled.
        :param message: The message to log
        """
        if self.verbose:
            print(message)

    async def acall(self, input: Any, verification_fn: Optional[Callable[[Any], None]] = None) -> Any:
        async with self.semaphore:
            self._log("Acquiring semaphore and making a call")
            try:
                result = await self.chain.acall(input)
                if verification_fn:
                    verification_fn(result)
                self._log("Call succeeded")
                return result
            except Exception as e:
                self._log(f"Call failed: {str(e)}")
                raise


class ParallelRetryLLMChain:
    def __init__(self, chain, n_parallel_retries=4, n_sequential_retries=40, max_sleep_s=10, max_sleep_safety_s=None, verbose=False):
        """
        Initializes the ParallelRetryLLMChain.
        :param chain: The chain to be executed
        :param n_parallel_retries: Number of parallel retries
        :param n_sequential_retries: Number of sequential retries
        :param max_sleep_s: Maximum sleep time between retries
        :param max_sleep_safety_s: Maximum sleep time before each call for safety
        :param verbose: Flag to enable/disable verbose logging
        """
        self.chain = chain
        self.n_parallel_retries = n_parallel_retries
        self.n_sequential_retries = n_sequential_retries
        self.max_sleep_s = max_sleep_s
        self.max_sleep_safety_s = max_sleep_safety_s or n_parallel_retries
        self.verbose = verbose

    def _log(self, message):
        """
        Logs a message if verbose is enabled.
        :param message: The message to log
        """
        if self.verbose:
            print(message)

    async def _make_call_with_sleep(self, input):
        await asyncio.sleep(random.uniform(0, self.max_sleep_safety_s))
        return await self.chain.acall(input)

    async def acall(self, input, verification_fn=None):
        for attempt in range(self.n_sequential_retries):
            self._log(
                f"Sequential attempt {attempt + 1}/{self.n_sequential_retries}")

            tasks = [asyncio.create_task(self._make_call_with_sleep(
                input)) for _ in range(self.n_parallel_retries)]
            while tasks:
                done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    try:
                        result = await task
                        result_text = result['text']
                        if verification_fn:
                            verification_fn(result_text)
                        self._log("Parallel success")
                        return result_text
                    except Exception as e:
                        self._log(f"Parallel attempt failed: {str(e)}")

            self._log("All parallel attempts failed in this sequential round")
            await asyncio.sleep(self.max_sleep_s)

        self._log("All retries failed")
        raise RuntimeError("All retries failed")
