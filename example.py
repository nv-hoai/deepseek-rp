import os
import sys
from collections.abc import Generator

from dotenv import load_dotenv

from dsk import (
    APIError,
    AuthenticationError,
    Chunk,
    DeepSeekClient,
    NetworkError,
    RateLimitError,
    WafError,
)

load_dotenv()


def print_response(chunks: Generator[Chunk, None, None]) -> None:
    thinking, search, text, title = [], [], [], ""
    for chunk in chunks:
        if chunk.type == "thinking":
            thinking.append(chunk.content)
        elif chunk.type == "search":
            search.append(chunk.content)
        elif chunk.type == "text":
            text.append(chunk.content)
        elif chunk.type == "title":
            title = chunk.content
    if thinking:
        print("\nThinking:")
        print("".join(thinking).strip())
        print()
    if search:
        print("\nSearch:")
        print("".join(search).strip())
        print()
    if title:
        print(f"Title: {title}\n")
    print("Response:")
    print("".join(text))
    print()


def run_chat_example(client: DeepSeekClient, title: str, prompt: str,
                     thinking_enabled: bool = True,
                     search_enabled: bool = False) -> None:
    print(f"\n{title}")
    print("-" * 80)
    try:
        print_response(client.chat_completion(
            client.create_chat_session(),
            prompt,
            thinking_enabled=thinking_enabled,
            search_enabled=search_enabled,
        ))
    except AuthenticationError:
        print("Authentication failed. Check your token.")
        sys.exit(1)
    except RateLimitError:
        print("Rate limited. Wait before retrying.")
    except WafError as e:
        print(f"WAF protection encountered: {e}")
    except NetworkError:
        print("Network error. Check your connection.")
    except APIError as e:
        print(f"API error: {e}")
        if e.status_code:
            print(f"Status code: {e.status_code}")


def main() -> None:
    try:
        # LocalStorage `userToken` -> value, or: python -m dsk.auth <email> <password>
        client = DeepSeekClient(os.getenv("DEEPSEEK_AUTH_TOKEN", ""))
        run_chat_example(client, "Example 3: Simple calculation (no thinking)",
                         "What is 2+2?", thinking_enabled=False)
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nFatal Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
