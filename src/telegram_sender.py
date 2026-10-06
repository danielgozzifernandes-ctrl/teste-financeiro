"""
Envia a mensagem de texto e o gráfico PNG para o Telegram via Bot API.

requests direto em vez de python-telegram-bot: só enviamos, não precisa
de bot interativo.

send_message() → /sendMessage (MarkdownV2); send_photo() → /sendPhoto.
Retry com backoff em erro de rede e 429.
Env: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID (negativo para grupos/canais).
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# URL base da Telegram Bot API
_BASE_URL = "https://api.telegram.org/bot{token}/{method}"

# Timeout por requisição (segundos)
_REQUEST_TIMEOUT = 30

# Número de tentativas e base do backoff
_MAX_RETRIES = 3
_BACKOFF_BASE = 2  # segundos: 2, 4, 8

# Limite de caracteres do Telegram por mensagem
_TELEGRAM_MSG_LIMIT   = 4096
_TELEGRAM_CAP_LIMIT   = 1024
# Nota adicionada ao final quando a mensagem é truncada pelo sender
_TRUNCATE_NOTICE = "\n\n⚠️ _\\[relatório truncado para caber no limite do Telegram\\]_"


def _truncate_message(text: str, limit: int) -> str:
    """
    Trunca texto para caber no limite do Telegram, adicionando nota visível.

    Preserva o máximo de conteúdo possível cortando pelo último caractere
    que cabe, depois acrescenta _TRUNCATE_NOTICE ao final.
    """
    if len(text) <= limit:
        return text
    notice = _TRUNCATE_NOTICE
    available = limit - len(notice)
    logger.warning(
        "Mensagem (%d chars) excede limite do Telegram (%d). Truncando.",
        len(text), limit,
    )
    return text[:available] + notice


class TelegramError(Exception):
    """Erro não-recuperável ao enviar para o Telegram."""


class TelegramSender:
    """
    Envia mensagens e fotos para um canal/grupo do Telegram.
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ) -> None:
        self.bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id   = chat_id   or os.environ.get("TELEGRAM_CHAT_ID", "")

        if not self.bot_token:
            raise TelegramError(
                "TELEGRAM_BOT_TOKEN não definido. "
                "Configure a variável de ambiente ou passe bot_token= no construtor."
            )
        if not self.chat_id:
            raise TelegramError(
                "TELEGRAM_CHAT_ID não definido. "
                "Configure a variável de ambiente ou passe chat_id= no construtor."
            )

    # API pública

    def send_report(
        self,
        text: str,
        chart_path: Optional[Path] = None,
        disable_notification: bool = False,
    ) -> None:
        """
        Envia o relatório completo: texto + gráfico (se disponível).

        Se chart_path for fornecido e existir, envia como foto com o texto
        como caption (limite de 1024 chars). Caso contrário, envia apenas o texto.

        Args:
            text:                 Mensagem em MarkdownV2 (≤4096 chars).
            chart_path:           Caminho para o PNG do gráfico (opcional).
            disable_notification: True para enviar silenciosamente.
        """
        if chart_path and chart_path.exists():
            self._send_photo_with_caption(
                text=text,
                photo_path=chart_path,
                disable_notification=disable_notification,
            )
        else:
            if chart_path:
                logger.warning("Gráfico não encontrado em %s — enviando só texto.", chart_path)
            self.send_message(text=text, disable_notification=disable_notification)

    def send_message(
        self,
        text: str,
        disable_notification: bool = False,
    ) -> dict:
        """
        POST /sendMessage com parse_mode=MarkdownV2.

        Trunca automaticamente se o texto exceder 4096 chars, adicionando
        nota visível ao leitor. Preferível a receber HTTP 400 da API.

        Returns:
            Resposta JSON do Telegram em caso de sucesso.

        Raises:
            TelegramError: após _MAX_RETRIES tentativas falhas.
        """
        safe_text = _truncate_message(text, _TELEGRAM_MSG_LIMIT)
        payload = {
            "chat_id":              self.chat_id,
            "text":                 safe_text,
            "parse_mode":           "MarkdownV2",
            "disable_notification": disable_notification,
        }
        return self._post("sendMessage", data=payload)

    def send_photo(
        self,
        photo_path: Path,
        caption: Optional[str] = None,
        disable_notification: bool = False,
    ) -> dict:
        """
        POST /sendPhoto com arquivo PNG.

        Args:
            photo_path:           Caminho para o arquivo PNG.
            caption:              Legenda em MarkdownV2 (≤1024 chars, opcional).
            disable_notification: True para enviar silenciosamente.

        Returns:
            Resposta JSON do Telegram.

        Raises:
            TelegramError: se o arquivo não existir ou após retries esgotados.
        """
        if not photo_path.exists():
            raise TelegramError(f"Arquivo de gráfico não encontrado: {photo_path}")

        data = {
            "chat_id":              self.chat_id,
            "disable_notification": disable_notification,
        }
        if caption:
            data["caption"]    = caption
            data["parse_mode"] = "MarkdownV2"

        with open(photo_path, "rb") as f:
            return self._post("sendPhoto", data=data, files={"photo": f})

    # Estratégia de envio foto+caption

    def _send_photo_with_caption(
        self,
        text: str,
        photo_path: Path,
        disable_notification: bool,
    ) -> None:
        """
        Envia foto com caption (≤1024) + mensagem completa como reply se necessário.

        Telegram limita caption a 1024 chars. Se o texto exceder isso,
        envia a foto sem caption e depois o texto completo como mensagem separada.
        """
        _CAPTION_LIMIT = 1024

        if len(text) <= _CAPTION_LIMIT:
            self.send_photo(
                photo_path=photo_path,
                caption=text,
                disable_notification=disable_notification,
            )
        else:
            # Foto sem caption + mensagem completa separada
            logger.debug(
                "Texto (%d chars) excede caption limit. Enviando foto + mensagem separada.",
                len(text),
            )
            self.send_photo(
                photo_path=photo_path,
                disable_notification=disable_notification,
            )
            self.send_message(text=text, disable_notification=disable_notification)

    # HTTP com retry

    def _post(
        self,
        method: str,
        data: dict,
        files: Optional[dict] = None,
    ) -> dict:
        """
        Executa POST para a Bot API com retry exponencial.

        Retenta em:
          - Exceções de rede (ConnectionError, Timeout)
          - HTTP 429 (rate limit) — respeita Retry-After se presente
          - HTTP 5xx (erro do servidor do Telegram)

        Não retenta em:
          - HTTP 4xx (erro nosso — token inválido, chat_id errado, etc.)
        """
        url = _BASE_URL.format(token=self.bot_token, method=method)

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    url,
                    data=data,
                    files=files,
                    timeout=_REQUEST_TIMEOUT,
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                wait = _BACKOFF_BASE ** attempt
                logger.warning(
                    "Tentativa %d/%d falhou (%s). Aguardando %ds.",
                    attempt, _MAX_RETRIES, exc, wait,
                )
                if attempt == _MAX_RETRIES:
                    raise TelegramError(f"Falha de rede após {_MAX_RETRIES} tentativas: {exc}") from exc
                time.sleep(wait)
                continue

            # Rate limit
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", _BACKOFF_BASE ** attempt))
                logger.warning(
                    "Rate limit (429). Aguardando %ds (tentativa %d/%d).",
                    retry_after, attempt, _MAX_RETRIES,
                )
                if attempt == _MAX_RETRIES:
                    raise TelegramError("Rate limit persistente após retries.")
                time.sleep(retry_after)
                continue

            # Erros do servidor — retentável
            if resp.status_code >= 500:
                wait = _BACKOFF_BASE ** attempt
                logger.warning(
                    "Erro %d do servidor Telegram. Aguardando %ds (tentativa %d/%d).",
                    resp.status_code, wait, attempt, _MAX_RETRIES,
                )
                if attempt == _MAX_RETRIES:
                    raise TelegramError(f"Servidor Telegram com erro {resp.status_code} após retries.")
                time.sleep(wait)
                continue

            # Erro do cliente (4xx) — não retenta
            if not resp.ok:
                try:
                    body = resp.json()
                    description = body.get("description", resp.text)
                except Exception:
                    description = resp.text
                raise TelegramError(
                    f"Telegram API erro {resp.status_code}: {description}"
                )

            # Sucesso
            result = resp.json()
            logger.debug("Telegram API %s OK (attempt %d).", method, attempt)
            return result

        # Nunca deve chegar aqui (loop garante raise antes)
        raise TelegramError("Retries esgotados sem resultado.")

    # Diagnóstico

    def get_me(self) -> dict:
        """Chama /getMe para validar o token. Útil em testes e dry-run."""
        return self._post("getMe", data={})


# Função de conveniência

def send_report(
    text: str,
    chart_path: Optional[Path] = None,
    bot_token: Optional[str] = None,
    chat_id: Optional[str] = None,
    disable_notification: bool = False,
) -> None:
    """
    Ponto de entrada simplificado para main.py.

    Raises:
        TelegramError: se o envio falhar após retries.
    """
    sender = TelegramSender(bot_token=bot_token, chat_id=chat_id)
    sender.send_report(
        text=text,
        chart_path=chart_path,
        disable_notification=disable_notification,
    )
