import json
import os
from dataclasses import dataclass
from pathlib import Path

SETTINGS_PATH = Path(os.getenv("SETTINGS_PATH", "/app/settings.json"))


@dataclass(frozen=True)
class CommonSettings:
    model_name: str
    split_layer: int
    dtype: str


@dataclass(frozen=True)
class ServerSettings:
    host: str
    port: int


@dataclass(frozen=True)
class ClientSettings:
    server_host: str
    port: int
    max_new_tokens: int
    use_chat_template: bool
    system_prompt: str


@dataclass(frozen=True)
class AppSettings:
    common: CommonSettings
    server: ServerSettings
    client: ClientSettings


_DEFAULT = {
    "common": {
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "split_layer": 12,
        "dtype": "float32",
    },
    "server": {"host": "0.0.0.0", "port": 65432},
    "client": {
        "server_host": "localhost",
        "port": 65432,
        "max_new_tokens": 30,
        "use_chat_template": True,
        "system_prompt": "You are a helpful assistant.",
    },
}


def _read_file_settings() -> dict:
    if not SETTINGS_PATH.exists():
        return _DEFAULT
    with SETTINGS_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    merged = {
        "common": {**_DEFAULT["common"], **data.get("common", {})},
        "server": {**_DEFAULT["server"], **data.get("server", {})},
        "client": {**_DEFAULT["client"], **data.get("client", {})},
    }
    return merged


def load_settings() -> AppSettings:
    data = _read_file_settings()

    data["common"]["model_name"] = os.getenv("MODEL_NAME", data["common"]["model_name"])
    data["common"]["split_layer"] = int(
        os.getenv("SPLIT_LAYER", data["common"]["split_layer"])
    )
    data["common"]["dtype"] = os.getenv("MODEL_DTYPE", data["common"]["dtype"])

    data["server"]["host"] = os.getenv("SERVER_BIND_HOST", data["server"]["host"])
    data["server"]["port"] = int(os.getenv("PORT", data["server"]["port"]))

    data["client"]["server_host"] = os.getenv(
        "SERVER_HOST", data["client"]["server_host"]
    )
    data["client"]["port"] = int(os.getenv("PORT", data["client"]["port"]))
    data["client"]["max_new_tokens"] = int(
        os.getenv("MAX_NEW_TOKENS", data["client"]["max_new_tokens"])
    )
    data["client"]["use_chat_template"] = os.getenv(
        "USE_CHAT_TEMPLATE", str(data["client"]["use_chat_template"])
    ).strip().lower() in {"1", "true", "yes", "on"}
    data["client"]["system_prompt"] = os.getenv(
        "SYSTEM_PROMPT", data["client"]["system_prompt"]
    )

    return AppSettings(
        common=CommonSettings(**data["common"]),
        server=ServerSettings(**data["server"]),
        client=ClientSettings(**data["client"]),
    )
