"""Доступ запланированной задачи к обращениям без вывода служебного токена."""

import argparse
import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def main() -> None:
    parser = argparse.ArgumentParser(description="Прочитать или обновить обращения Гровики")
    parser.add_argument("--api-url", default=os.getenv("MLSYSTEM2_FEEDBACK_API_URL", "http://127.0.0.1:8091/api/v1"))
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="Прочитать обращения")
    listing.add_argument("--active", action="store_true", help="Только незавершённые обращения")
    detail = commands.add_parser("get", help="Прочитать одно обращение")
    detail.add_argument("id", type=int)
    change = commands.add_parser("update", help="Обновить обращение JSON-запросом из стандартного ввода")
    change.add_argument("id", type=int)
    args = parser.parse_args()
    token = os.getenv("MLSYSTEM2_FEEDBACK_API_TOKEN", "")
    if not token:
        parser.error("Не настроен служебный доступ к обращениям")
    base = args.api_url.rstrip("/") + "/feedback"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        if args.command == "list":
            items, cursor = [], ""
            while True:
                url = f"{base}?limit=100&active={'true' if args.active else 'false'}{cursor}"
                with urlopen(Request(url, headers=headers), timeout=30) as response:
                    page = json.load(response)
                items.extend(page["items"])
                if not page["has_more"]:
                    result = {"items": items}
                    break
                cursor = f"&before_id={page['items'][-1]['id']}"
        else:
            body = json.dumps(json.load(sys.stdin), ensure_ascii=False).encode("utf-8") if args.command == "update" else None
            with urlopen(Request(f"{base}/{args.id}", data=body, headers=headers, method="PATCH" if body is not None else "GET"), timeout=30) as response:
                result = json.load(response)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (HTTPError, URLError, ValueError) as error:
        detail = error.read().decode("utf-8", errors="replace") if isinstance(error, HTTPError) else str(error)
        print(f"Не удалось обработать обращение: {detail}", file=sys.stderr)
        raise SystemExit(1) from error
