import logging

aemlogger = logging.getLogger("")


def configure_logging(verbosity):
    aemlogger.setLevel(verbosity)
    ch = logging.StreamHandler()
    ch.setFormatter(ElapsedFormatter())

    aemlogger.addHandler(ch)


class ElapsedFormatter(logging.Formatter):
    """Format logging message to include elapsed time."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        """Format incoming message."""
        lvl = record.levelname
        name = record.name
        t = int(round(record.relativeCreated / 1000.0))
        msg = record.getMessage()
        logstr = "+{}s {}:{} {}".format(t, name, lvl, msg)
        return logstr
