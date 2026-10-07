"""Configuration loading.

Two sources, kept separate on purpose:

* ``.env`` (+ process environment) — mode flags and secrets. Loaded by ``EnvSettings``.
* ``config/settings.yaml`` — pairs, schedule, paper-broker and risk parameters, models.
  Loaded by ``TradingSettings``.

Risk limits live here and in code, never in an LLM prompt (Safety Invariant #3).
Validation is strict: unknown keys are rejected, so a typo in a risk limit fails loudly
instead of silently falling back to a default.
"""

from __future__ import annotations

import re
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_ENV_FILE = Path(".env")
DEFAULT_SETTINGS_PATH = Path("config/settings.yaml")

# Spot symbols only, e.g. "BTC/CAD". ccxt derivative symbols ("BTC/USD:USD") are rejected
# (Safety Invariant #5).
_SPOT_PAIR_RE = re.compile(r"^[A-Z0-9]+/[A-Z0-9]+$")
_ACCOUNT_NAME_RE = r"^[a-z0-9][a-z0-9_-]*$"

BenchmarkName = Literal["buy_and_hold", "ma_crossover", "do_nothing"]
Percent = Annotated[Decimal, Field(gt=0, le=100)]
FeePercent = Annotated[Decimal, Field(ge=0, lt=100)]


class ConfigError(Exception):
    """Configuration is missing or invalid. The bot must not start."""


class Mode(StrEnum):
    PAPER = "paper"
    LIVE = "live"


# --------------------------------------------------------------------------- .env


class EnvSettings(BaseSettings):
    """Mode flags and secrets from ``.env`` / the process environment.

    Process environment variables take precedence over the ``.env`` file.
    Secrets are ``SecretStr`` so they never appear in reprs or logs (Safety Invariant #7).
    """

    model_config = SettingsConfigDict(
        env_file=DEFAULT_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    mode: Mode = Mode.PAPER
    # Interpreted only by the mode guard in ``ai_trader.main``.
    live_trading_confirmed: str = "no"

    kraken_api_key: SecretStr | None = None
    kraken_api_secret: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    telegram_bot_token: SecretStr | None = None

    telegram_chat_id: str | None = None
    healthcheck_url: SecretStr | None = None  # the ping URL embeds a secret UUID
    database_url: str = "sqlite:///data/trader.db"
    settings_path: Path = DEFAULT_SETTINGS_PATH

    @field_validator("mode", mode="before")
    @classmethod
    def _normalize_mode(cls, value: Any) -> Any:
        # Unset or empty MODE means paper (Safety Invariant #1).
        if value is None or (isinstance(value, str) and not value.strip()):
            return Mode.PAPER
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator(
        "kraken_api_key",
        "kraken_api_secret",
        "anthropic_api_key",
        "openai_api_key",
        "gemini_api_key",
        "telegram_bot_token",
        "telegram_chat_id",
        "healthcheck_url",
        mode="before",
    )
    @classmethod
    def _empty_to_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("live_trading_confirmed", mode="before")
    @classmethod
    def _none_to_no(cls, value: Any) -> Any:
        return "no" if value is None else value


# --------------------------------------------------------------------- settings.yaml


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PaperSettings(_StrictModel):
    starting_cash_cad: Annotated[Decimal, Field(gt=0)]
    taker_fee_pct: FeePercent
    maker_fee_pct: FeePercent


class RiskSettings(_StrictModel):
    max_trade_pct_of_equity: Percent
    max_position_pct_per_pair: Percent
    max_total_exposure_pct: Percent
    daily_loss_limit_pct: Percent
    max_drawdown_halt_pct: Percent
    min_minutes_between_trades: Annotated[int, Field(ge=0)]
    min_confidence: Annotated[Decimal, Field(ge=0, le=1)]
    max_trades_per_day: Annotated[int, Field(ge=1)]
    default_stop_loss_pct: Annotated[Decimal, Field(gt=0, lt=100)]

    @model_validator(mode="after")
    def _limits_are_consistent(self) -> RiskSettings:
        if self.max_trade_pct_of_equity > self.max_position_pct_per_pair:
            raise ValueError("max_trade_pct_of_equity must be <= max_position_pct_per_pair")
        if self.max_position_pct_per_pair > self.max_total_exposure_pct:
            raise ValueError("max_position_pct_per_pair must be <= max_total_exposure_pct")
        return self


class LLMSettings(_StrictModel):
    timeout_seconds: Annotated[int, Field(ge=1, le=600)]
    max_tokens: Annotated[int, Field(ge=16)]  # includes reasoning tokens on some providers
    max_daily_cost_usd: Annotated[Decimal, Field(gt=0)]  # per account; cycles hold beyond it


class ModelSettings(_StrictModel):
    name: Annotated[str, Field(pattern=_ACCOUNT_NAME_RE, max_length=32)]
    litellm_model: Annotated[str, Field(min_length=1)]
    # Optional per-model overrides. Leave temperature unset for reasoning models that
    # reject it.
    temperature: Annotated[float, Field(ge=0, le=2)] | None = None
    max_tokens: Annotated[int, Field(ge=16)] | None = None
    api_base: str | None = None  # e.g. a local Ollama server

    @property
    def is_placeholder(self) -> bool:
        """True while the model ID is still a template like ``anthropic/<model-id>``."""
        return "<" in self.litellm_model or ">" in self.litellm_model


class TradingSettings(_StrictModel):
    pairs: Annotated[list[str], Field(min_length=1)]
    quote_currency: Annotated[str, Field(pattern=r"^[A-Z]{3,5}$")]
    decision_interval_minutes: Annotated[int, Field(ge=1)]
    max_data_age_seconds: Annotated[int, Field(ge=1)]
    paper: PaperSettings
    risk: RiskSettings
    llm: LLMSettings
    models: list[ModelSettings] = Field(default_factory=list)
    benchmarks: list[BenchmarkName] = Field(default_factory=list)

    @field_validator("pairs", mode="before")
    @classmethod
    def _normalize_pairs(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [p.strip().upper() if isinstance(p, str) else p for p in value]
        return value

    @field_validator("pairs")
    @classmethod
    def _spot_pairs_only(cls, pairs: list[str]) -> list[str]:
        for pair in pairs:
            if not _SPOT_PAIR_RE.match(pair):
                raise ValueError(f"{pair!r} is not a spot pair like 'BTC/CAD'")
        if len(set(pairs)) != len(pairs):
            raise ValueError("pairs must be unique")
        return pairs

    @model_validator(mode="after")
    def _cross_checks(self) -> TradingSettings:
        wrong_quote = [p for p in self.pairs if p.split("/")[1] != self.quote_currency]
        if wrong_quote:
            raise ValueError(f"pairs {wrong_quote} are not quoted in {self.quote_currency}")

        model_names = [m.name for m in self.models]
        if len(set(model_names)) != len(model_names):
            raise ValueError("model names must be unique")
        if len(set(self.benchmarks)) != len(self.benchmarks):
            raise ValueError("benchmarks must be unique")
        clash = set(model_names) & set(self.benchmarks)
        if clash:
            raise ValueError(f"model names clash with benchmark names: {sorted(clash)}")
        return self


# --------------------------------------------------------------------------- loading


class AppConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    env: EnvSettings
    trading: TradingSettings


def load_env(env_file: Path | str | None = DEFAULT_ENV_FILE) -> EnvSettings:
    """Load ``EnvSettings``. ``env_file=None`` reads only the process environment."""
    try:
        return EnvSettings(_env_file=env_file)  # type: ignore[call-arg]
    except ValidationError as exc:
        # include_input=False: never echo raw values, which could be secrets.
        details = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']).upper()}: {err['msg']}"
            for err in exc.errors(include_input=False, include_url=False)
        )
        raise ConfigError(f"Invalid environment configuration: {details}") from None


def load_trading_settings(path: Path | str) -> TradingSettings:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"Settings file not found: {path}") from None
    except yaml.YAMLError as exc:
        raise ConfigError(f"Settings file {path} is not valid YAML: {exc}") from None
    if not isinstance(raw, dict):
        raise ConfigError(f"Settings file {path} must contain a mapping at the top level")
    try:
        return TradingSettings.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid settings in {path}:\n{exc}") from None


def load_config(
    env_file: Path | str | None = DEFAULT_ENV_FILE,
    settings_path: Path | str | None = None,
) -> AppConfig:
    """Load and validate all configuration. Raises ``ConfigError`` on any problem."""
    env = load_env(env_file)
    trading = load_trading_settings(settings_path or env.settings_path)
    return AppConfig(env=env, trading=trading)
