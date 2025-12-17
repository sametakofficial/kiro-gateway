# -*- coding: utf-8 -*-

# Kiro OpenAI Gateway
# Copyright (C) 2025 Jwadow
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Kiro API Gateway - OpenAI-compatible interface for Kiro API.

Application entry point. Creates FastAPI app and connects routes.

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000
    or directly:
    python main.py
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from kiro_gateway.config import (
    APP_TITLE,
    APP_DESCRIPTION,
    APP_VERSION,
    REFRESH_TOKEN,
    PROFILE_ARN,
    REGION,
    KIRO_CREDS_FILE,
    PROXY_API_KEY,
    LOG_LEVEL,
    _warn_deprecated_debug_setting,
)
from kiro_gateway.auth import KiroAuthManager
from kiro_gateway.cache import ModelInfoCache
from kiro_gateway.routes import router
from kiro_gateway.exceptions import validation_exception_handler


# --- Loguru Configuration ---
logger.remove()
logger.add(
    sys.stderr,
    level=LOG_LEVEL,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)


class InterceptHandler(logging.Handler):
    """
    Перехватывает логи из стандартного logging и перенаправляет в loguru.
    
    Это позволяет захватывать логи uvicorn, FastAPI и других библиотек,
    которые используют стандартный logging вместо loguru.
    """
    
    def emit(self, record: logging.LogRecord) -> None:
        # Получаем соответствующий уровень loguru
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        
        # Находим вызывающий фрейм для корректного отображения источника
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging_intercept():
    """
    Настраивает перехват логов из стандартного logging в loguru.
    
    Перехватывает логи от:
    - uvicorn (access logs, error logs)
    - uvicorn.error
    - uvicorn.access
    - fastapi
    """
    # Список логгеров для перехвата
    loggers_to_intercept = [
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "fastapi",
    ]
    
    for logger_name in loggers_to_intercept:
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler()]
        logging_logger.propagate = False


# Настраиваем перехват логов uvicorn/fastapi
setup_logging_intercept()


# --- Configuration Validation ---
def validate_configuration() -> None:
    """
    Validates that required configuration is present.
    
    Checks:
    - .env file exists
    - Either REFRESH_TOKEN or KIRO_CREDS_FILE is configured
    
    Raises:
        SystemExit: If critical configuration is missing
    """
    errors = []
    
    # Check if .env file exists
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists():
        errors.append(
            ".env file not found!\n"
            "\n"
            "To get started:\n"
            "1. Create .env or rename from .env.example:\n"
            "   cp .env.example .env\n"
            "\n"
            "2. Edit .env and configure your credentials:\n"
            "   2.1. Set you super-secret password as PROXY_API_KEY\n"
            "   2.2. Set your Kiro credentials:\n"
            "      - 1 way: KIRO_CREDS_FILE to your Kiro credentials JSON file\n"
            "      - 2 way: REFRESH_TOKEN from Kiro IDE traffic\n"
            "\n"
            "See README.md for detailed instructions."
        )
    else:
        # .env exists, check for credentials
        has_refresh_token = bool(REFRESH_TOKEN)
        has_creds_file = bool(KIRO_CREDS_FILE)
        
        # Check if creds file actually exists
        if KIRO_CREDS_FILE:
            creds_path = Path(KIRO_CREDS_FILE).expanduser()
            if not creds_path.exists():
                has_creds_file = False
                logger.warning(f"KIRO_CREDS_FILE not found: {KIRO_CREDS_FILE}")
        
        if not has_refresh_token and not has_creds_file:
            errors.append(
                "No Kiro credentials configured!\n"
                "\n"
                "   Configure one of the following in your .env file:\n"
                "\n"
                "Set you super-secret password as PROXY_API_KEY\n"
                "   PROXY_API_KEY=\"my-super-secret-password-123\"\n"
                "\n"
                "   Option 1 (Recommended): JSON credentials file\n"
                "      KIRO_CREDS_FILE=\"path/to/your/kiro-credentials.json\"\n"
                "\n"
                "   Option 2: Refresh token\n"
                "      REFRESH_TOKEN=\"your_refresh_token_here\"\n"
                "\n"
                "   See README.md for how to obtain credentials."
            )
    
    # Print errors and exit if any
    if errors:
        logger.error("")
        logger.error("=" * 60)
        logger.error("  CONFIGURATION ERROR")
        logger.error("=" * 60)
        for error in errors:
            for line in error.split('\n'):
                logger.error(f"  {line}")
        logger.error("=" * 60)
        logger.error("")
        sys.exit(1)
    
    # Log successful configuration
    if KIRO_CREDS_FILE:
        logger.info(f"Using credentials file: {KIRO_CREDS_FILE}")
    elif REFRESH_TOKEN:
        logger.info("Using refresh token from environment")


# Run configuration validation on import
validate_configuration()

# Warn about deprecated DEBUG_LAST_REQUEST if used
_warn_deprecated_debug_setting()


# --- Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Управляет жизненным циклом приложения.
    
    Создаёт и инициализирует:
    - KiroAuthManager для управления токенами
    - ModelInfoCache для кэширования моделей
    """
    logger.info("Starting application... Creating state managers.")
    
    # Создаём AuthManager
    app.state.auth_manager = KiroAuthManager(
        refresh_token=REFRESH_TOKEN,
        profile_arn=PROFILE_ARN,
        region=REGION,
        creds_file=KIRO_CREDS_FILE if KIRO_CREDS_FILE else None
    )
    
    # Создаём кэш моделей
    app.state.model_cache = ModelInfoCache()
    
    yield
    
    logger.info("Shutting down application.")


# --- FastAPI приложение ---
app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan
)


# --- CORS Middleware ---
# Разрешаем CORS для всех источников, чтобы поддерживать браузерные клиенты
# и инструменты, которые отправляют preflight OPTIONS запросы
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешаем все источники
    allow_credentials=True,
    allow_methods=["*"],  # Разрешаем все методы (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Разрешаем все заголовки
)


# --- Регистрация обработчика ошибок валидации ---
app.add_exception_handler(RequestValidationError, validation_exception_handler)


# --- Подключение роутов ---
app.include_router(router)


# --- Uvicorn log config ---
# Минимальная конфигурация для перенаправления логов uvicorn в loguru.
# Использует InterceptHandler, который перехватывает логи и передаёт их в loguru.
UVICORN_LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "default": {
            "class": "main.InterceptHandler",
        },
    },
    "loggers": {
        "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.access": {"handlers": ["default"], "level": "INFO", "propagate": False},
    },
}


# --- Точка входа ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server...")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_config=UVICORN_LOG_CONFIG,
    )
