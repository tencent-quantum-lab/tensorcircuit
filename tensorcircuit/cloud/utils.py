"""
utility functions for cloud connection
"""

from typing import Any, Callable, Optional
from functools import wraps
import inspect
import logging
import os
import sys
import time

import requests

# from simplejson.errors import JSONDecodeError

logger = logging.getLogger(__name__)
thismodule = sys.modules[__name__]


class HttpStatusError(Exception):
    """
    Used when the return request has http code beyond 200
    """

    pass


# TODO(@refraction-ray): whether an exception hierarchy for tc is necessary?
connection_errors = (
    ConnectionResetError,
    HttpStatusError,
    requests.exceptions.RequestException,
    requests.exceptions.ConnectionError,
    requests.exceptions.SSLError,
    ValueError,
    # JSONDecodeError,
)


def set_proxy(proxy: Optional[str] = None) -> None:
    """
    :param proxy: str. format as "http://user:passwd@host:port" user passwd part can be omitted if not set.
        None for turning off the proxy.
    :return:
    """
    if proxy:
        os.environ["http_proxy"] = proxy
        os.environ["https_proxy"] = proxy
        setattr(thismodule, "proxy", proxy)
    else:
        os.environ["http_proxy"] = ""
        os.environ["https_proxy"] = ""
        setattr(thismodule, "proxy", None)


def reconnect(tries: int = 5, timeout: int = 12) -> Callable[..., Any]:
    # wrapper originally designed in xalpha by @refraction-ray
    # https://github.com/refraction-ray/xalpha
    def robustify(f: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(f)
        def wrapper(*args: Any, **kws: Any) -> Any:
            if getattr(thismodule, "proxy", None):
                kws["proxies"] = {
                    "http": getattr(thismodule, "proxy"),
                    "https": getattr(thismodule, "proxy"),
                }
                logger.debug("Using proxy %s" % getattr(thismodule, "proxy"))
            kws["timeout"] = timeout
            if args:
                url = args[0]
            else:
                url = kws.get("url", "")
            headers = kws.get("headers", {})
            if (not headers.get("user-agent", None)) and (
                not headers.get("User-Agent", None)
            ):
                headers["user-agent"] = "Mozilla/5.0"
            kws["headers"] = headers
            for count in range(tries):
                try:
                    logger.debug(
                        "Fetching url: %s . Inside function `%s`"
                        % (url, inspect.stack()[1].function)
                    )
                    r = f(*args, **kws)
                    if (
                        getattr(r, "status_code", 200) != 200
                    ):  # in case r is a json dict
                        raise HttpStatusError
                    return r
                except connection_errors as e:
                    logger.warning("Fails at fetching url: %s. Try again." % url)
                    if count == tries - 1:
                        logger.error(
                            "Still wrong at fetching url: %s. after %s tries."
                            % (url, tries)
                        )
                        logger.error("Fails due to %s" % e.args[0])
                        raise e
                    time.sleep(0.5 * count)

        return wrapper

    return robustify


rget = reconnect()(requests.get)
rpost = reconnect()(requests.post)


@reconnect()
def rget_json(*args: Any, **kws: Any) -> Any:
    r = requests.get(*args, **kws)
    return r.json()


@reconnect()
def rpost_json(*args: Any, **kws: Any) -> Any:
    r = requests.post(*args, **kws)
    return r.json()
