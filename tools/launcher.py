import os
import subprocess
import sys
import traceback
from pathlib import Path

try:
    from rich.box import DOUBLE
    from rich.console import Console
    from rich.panel import Panel
except ImportError:
    Console = None
    Panel = None
    DOUBLE = None

try:
    import questionary
except ImportError:
    questionary = None


ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "tools" / "predict.py"
DEFAULT_IMAGE = str(ROOT / "samples" / "test.png")
DEFAULT_TASK = "ocr"
DEFAULT_TOKENS = 64
DEFAULT_PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
    "spotting": "Spotting:",
    "seal": "Seal Recognition:",
}


class CLI:
    def __init__(self):
        self.console = Console() if Console else None

    def print_title(self):
        os.system("cls" if os.name == "nt" else "clear")
        if self.console and Panel:
            self.console.print(
                Panel.fit(
                    "[bold yellow]Jugo project[/bold yellow]\n"
                    "[italic]Interactive Launcher pipeline[/italic]",
                    box=DOUBLE,
                    title="[bold red]Welcome[/bold red]",
                    subtitle="[bold red]Group 7[/bold red]",
                    padding=(1, 8),
                    border_style="bright_red",
                ),
                justify="center",
            )
        else:
            print("=== PaddleOCR-VL Interactive Launcher ===")

    def print_info(self, message):
        if self.console:
            self.console.print(f"[green]{message}[/green]")
        else:
            print(message)

    def print_error(self, error):
        if self.console:
            self.console.print("[bold red]Error occurred:[/bold red]", style="red")
            self.console.print(f"[red]{str(error)}[/red]")
            self.console.print(f"[red]Error type: {type(error).__name__}[/red]")
            if hasattr(error, "__cause__") and error.__cause__ is not None:
                self.console.print(f"[red]Caused by: {error.__cause__}[/red]")
            self.console.print("[red]Traceback:[/red]")
            self.console.print(f"[red]{traceback.format_exc()}[/red]")
        else:
            print(f"Error: {error}")
            print(traceback.format_exc())

    def print_shutdown(self):
        if self.console:
            self.console.print("\n[yellow]Shutting down...[/yellow]")
        else:
            print("\nShutting down...")

    def ask_text(self, message, default="", allow_empty=False):
        if questionary:
            answer = questionary.text(
                message,
                default=default,
                validate=lambda text: text != "" if not allow_empty else True,
            ).ask()
            return answer if answer != "" else None

        prompt = f"{message}"
        if default:
            prompt += f" [{default}]"
        prompt += ": "
        answer = input(prompt).strip()
        if answer:
            return answer
        if default:
            return default
        if allow_empty:
            return None
        raise ValueError(f"{message} cannot be empty.")

    def ask_select(self, message, choices, default=None):
        if questionary:
            return questionary.select(message, choices=choices, default=default).ask()

        print(f"\n{message}")
        for index, choice in enumerate(choices, start=1):
            marker = " (default)" if choice == default else ""
            print(f"  [{index}] {choice}{marker}")
        raw = input("Choose a number or press Enter for default: ").strip()
        if not raw:
            if default is not None:
                return default
            return choices[0]
        selected = int(raw)
        if selected < 1 or selected > len(choices):
            raise ValueError(f"Invalid selection: {raw}")
        return choices[selected - 1]

    def ask_number(self, message, default):
        if questionary:
            answer = questionary.text(
                message,
                default=str(default),
                validate=lambda value: value.isdigit() and int(value) > 0,
            ).ask()
            return int(answer)

        raw = input(f"{message} [{default}]: ").strip()
        if not raw:
            return default
        if not raw.isdigit() or int(raw) <= 0:
            raise ValueError(f"Invalid positive integer: {raw}")
        return int(raw)


def resolve_image_path(image_value: str) -> str:
    path = Path(image_value)
    if not path.is_absolute():
        path = ROOT / path
    return str(path.resolve())


def build_command(python_path: str, image_path: str, task: str, max_new_tokens: int, prompt: str):
    return [
        python_path,
        str(RUNNER),
        image_path,
        "--task",
        task,
        "--max-new-tokens",
        str(max_new_tokens),
        "--prompt",
        prompt,
    ]


def main() -> int:
    cli = CLI()
    try:
        python_default = sys.executable or r"c:\python314\python.exe"
        cli.print_title()
        python_path = cli.ask_text("Python executable", default=python_default)
        task = cli.ask_select("Choose task", list(DEFAULT_PROMPTS.keys()), default=DEFAULT_TASK)
        max_new_tokens = cli.ask_number("Token length", default=DEFAULT_TOKENS)
        prompt_mode = cli.ask_select(
            "Choose prompt mode",
            [
                f"Use default prompt ({DEFAULT_PROMPTS[task]})",
                "Enter custom prompt",
            ],
            default=f"Use default prompt ({DEFAULT_PROMPTS[task]})",
        )
        if prompt_mode == "Enter custom prompt":
            prompt = cli.ask_text("Prompt")
        else:
            prompt = DEFAULT_PROMPTS[task]

        image_mode = cli.ask_select(
            "Choose image",
            [f"Use default image ({DEFAULT_IMAGE})", "Enter custom image path"],
            default=f"Use default image ({DEFAULT_IMAGE})",
        )
        if image_mode == "Enter custom image path":
            image_path = resolve_image_path(cli.ask_text("Image path"))
        else:
            image_path = resolve_image_path(DEFAULT_IMAGE)

        if not Path(python_path).exists():
            raise FileNotFoundError(f"Python executable not found: {python_path}")
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        cli.print_info("Starting generation with:")
        cli.print_info(f"Task: {task}")
        cli.print_info(f"Prompt: {prompt}")
        cli.print_info(f"Max new tokens: {max_new_tokens}")
        cli.print_info(f"Image: {image_path}")

        command = build_command(python_path, image_path, task, max_new_tokens, prompt)
        completed = subprocess.run(command, cwd=str(ROOT))
        return completed.returncode
    except Exception as error:
        cli.print_error(error)
        return 1
    finally:
        cli.print_shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
