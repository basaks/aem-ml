import logging

aemlogger = logging.getLogger("")


def configure_logging(verbosity):
    aemlogger.setLevel(verbosity)
    ch = logging.StreamHandler()
    ch.setFormatter(ElapsedFormatter())

    aemlogger.addHandler(ch)
    """
    configures the logging of the verbosity file and the formatter is set
    """


class ElapsedFormatter(logging.Formatter):

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        """Format incoming message."""
        lvl = record.levelname
        name = record.name
        t = int(round(record.relativeCreated / 1000.0))
        msg = record.getMessage()
        logstr = "+{}s {}:{} {}".format(t, name, lvl, msg)
        return logstr

    """
    The logging message is formatted to include the elasped time taken for the
    logging process
    """
