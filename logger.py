import logging
import os


class Logger:
    def __init__(self, log_file_root_path, log_name, module_name="MOT"):
        super().__init__()

        self.logger = logging.getLogger(module_name)
        self.logger.setLevel(logging.INFO)
        log_file = os.path.join(log_file_root_path, '{}.log'.format(log_name))
        if not self.logger.handlers:
            fh = logging.FileHandler(log_file, encoding='utf-8')
            fh.setLevel(logging.INFO)

            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s -  %(threadName)s - %(process)d ")
            ch.setFormatter(formatter)
            fh.setFormatter(formatter)
            self.logger.addHandler(ch)
            self.logger.addHandler(fh)

    def error(self, msg, *args, **kwargs):
        if self.logger is not None:
            self.logger.error(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        if self.logger is not None:
            self.logger.info(msg, *args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        if self.logger is not None:
            self.logger.warning(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        if self.logger is not None:
            self.logger.warning(msg, *args, **kwargs)

    def exception(self, msg, *args, exc_info=True, **kwargs):
        if self.logger is not None:
            self.logger.exception(msg, *args, exc_info=True, **kwargs)
