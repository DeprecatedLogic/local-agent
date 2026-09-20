import logging
import sys

def setup_logging(level: int = logging.INFO) -> None:
    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)

    # Silence noisy HTTP/asyncio logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
