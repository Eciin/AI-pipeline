import argparse
import contextlib
import io
import itertools
import os
import sys
import threading
import time
import warnings
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.configuration_paddleocr_vl import PaddleOCRVLConfig
from model.modeling_paddleocr_vl import PaddleOCRVLForConditionalGeneration
from model.processing_paddleocr_vl import PaddleOCRVLProcessor


MODEL_PATH = ROOT / "model"
DEFAULT_IMAGE_PATH = ROOT / "samples" / "test.png"
DEFAULT_TASK = "ocr"
DEFAULT_MAX_NEW_TOKENS = 64
PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
    "spotting": "Spotting:",
    "seal": "Seal Recognition:",
}


class Spinner:
    def __init__(self, label: str):
        self.label = label
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        for frame in itertools.cycle(["|", "/", "-", "\\"]):
            if self._stop_event.is_set():
                break
            print(f"\r{self.label} {frame}", end="", flush=True)
            time.sleep(0.1)
        print("\r" + " " * (len(self.label) + 2) + "\r", end="", flush=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop_event.set()
        self._thread.join()
        return False


@contextlib.contextmanager
def capture_diagnostics():
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        with contextlib.redirect_stdout(stdout_buffer):
            with contextlib.redirect_stderr(stderr_buffer):
                yield stdout_buffer, stderr_buffer, recorded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PaddleOCR-VL on a local image.")
    parser.add_argument(
        "image_path",
        nargs="?",
        default=str(DEFAULT_IMAGE_PATH),
        help="Path to the input image. Defaults to test.png in this folder.",
    )
    parser.add_argument(
        "--task",
        choices=sorted(PROMPTS),
        default=DEFAULT_TASK,
        help="Recognition task prompt to use.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--prompt",
        help="Custom prompt text. Overrides the default prompt for the selected task.",
    )
    return parser.parse_args()


def get_runtime_config() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32


def load_image(image_path: str) -> Image.Image:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(path).convert("RGB")


def load_model(model_path: Path, device: str, dtype: torch.dtype):
    config = PaddleOCRVLConfig.from_pretrained(model_path)
    return (
        PaddleOCRVLForConditionalGeneration.from_pretrained(
            model_path,
            config=config,
            local_files_only=True,
            dtype=dtype,
            disable_progress_bar=True,
        )
        .to(device)
        .eval()
    )


def load_processor(model_path: Path):
    return PaddleOCRVLProcessor.from_pretrained(model_path, local_files_only=True)


def build_messages(image: Image.Image, prompt_text: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]


def move_to_device(batch: dict, device: str) -> dict:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def prepare_inputs(processor, messages: list[dict], device: str) -> dict:
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    return move_to_device(inputs, device)


def generate(model, inputs: dict, max_new_tokens: int) -> torch.Tensor:
    with torch.inference_mode():
        return model.generate(**inputs, max_new_tokens=max_new_tokens)


def decode_generated_text(processor, outputs: torch.Tensor, input_length: int) -> str:
    generated_tokens = outputs[0][input_length:]
    return processor.decode(generated_tokens, skip_special_tokens=True).strip()


def run_task(
    image_path: str, task: str, max_new_tokens: int, prompt_override: str | None = None
):
    prompt_text = prompt_override if prompt_override else PROMPTS[task]
    diagnostics = []

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    with capture_diagnostics() as (stdout_buffer, stderr_buffer, recorded_warnings):
        with Spinner("Loading image"):
            image = load_image(image_path)
            device, dtype = get_runtime_config()

        print(f"Image: {Path(image_path).resolve()}")
        print(f"Task: {task}")
        print(f"Prompt: {prompt_text}")
        print(f"Device: {device}")
        print(f"DType: {dtype}")

        start = time.perf_counter()
        with Spinner("Loading model"):
            model = load_model(MODEL_PATH, device, dtype)
            processor = load_processor(MODEL_PATH)
        print(f"Loaded model and processor in {time.perf_counter() - start:.1f}s")

        with Spinner("Preparing inputs"):
            messages = build_messages(image, prompt_text)
            inputs = prepare_inputs(processor, messages, device)

        start = time.perf_counter()
        with Spinner("Generating output"):
            outputs = generate(model, inputs, max_new_tokens)
        print(f"Generation finished in {time.perf_counter() - start:.1f}s")
        input_length = inputs["input_ids"].shape[-1]
        result = decode_generated_text(processor, outputs, input_length)

        stdout_text = stdout_buffer.getvalue().strip()
        stderr_text = stderr_buffer.getvalue().strip()
        if stdout_text:
            diagnostics.append(stdout_text)
        if stderr_text:
            diagnostics.append(stderr_text)
        if recorded_warnings:
            diagnostics.extend(
                f"{warning.category.__name__}: {warning.message}"
                for warning in recorded_warnings
            )

    return result, diagnostics


def main() -> None:
    args = parse_args()
    result, diagnostics = run_task(
        args.image_path,
        args.task,
        args.max_new_tokens,
        prompt_override=args.prompt,
    )
    print(f"Result: {result!r}")
    if diagnostics:
        print("\nDiagnostics:")
        seen = set()
        for entry in diagnostics:
            if entry not in seen:
                print(entry)
                seen.add(entry)


if __name__ == "__main__":
    main()
