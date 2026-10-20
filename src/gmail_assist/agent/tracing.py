"""Send ordinary log records to LangSmith, attached to the run that logged them.

With LangSmithLogHandler installed on the 'gmail_assist' logger, a
`log.info(...)` inside a graph node shows up in the LangSmith trace tree as
its own step under that node, named after the message (e.g. "INFO: local
model confidence 0.40 ..."), as well as on the console with -v. Outside a
traced run the handler does nothing.
"""
import logging

import langsmith as ls

# Step names in the trace tree get truncated beyond this; the full message
# is in the step's output.
MAX_NAME = 120


class LangSmithLogHandler(logging.Handler):
    def __init__(self, level: int = logging.INFO):
        super().__init__(level)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            parent = ls.get_current_run_tree()
            if parent is None:
                return
            message = record.getMessage()
            name = f'{record.levelname}: {message}'
            if len(name) > MAX_NAME:
                name = name[:MAX_NAME - 1] + '…'
            step = parent.create_child(
                name=name,
                run_type='chain',
                inputs={'logger': record.name, 'level': record.levelname},
                tags=['log', record.levelname.lower()],
            )
            step.end(outputs={'message': message})
            step.post()
        except Exception:
            self.handleError(record)
