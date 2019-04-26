#!/usr/bin/python
import logging
import logging.handlers

LOG_FILENAME = 'default.log'
LOG_TAG      = 'DeafultLogger'

class Logger:
    def __init__(self, logfile=LOG_FILENAME, logtag=LOG_TAG):
        self.logfile = logfile
        self.logobj  = logging.getLogger(logtag)
        self.logobj.setLevel(logging.DEBUG)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # create default console handler with a higher log level
        consolehandler = logging.StreamHandler()
        consolehandler.setFormatter(self.formatter)
        consolehandler.setLevel(logging.ERROR)
        self.logobj.addHandler(consolehandler)

    def setLevel(self, level):
        # Set up a specific logger with our desired output level
        self.logobj.setLevel(level)

    def setRotation(self):
        # Add the log message handler to the logger
        handler = logging.handlers.RotatingFileHandler(self.logfile, maxBytes=20000, backupCount=10)
        handler.setFormatter(self.formatter)
        self.logobj.addHandler(handler)

    def log(self, msg):
        # Log messages
        self.logobj.debug(msg)

if __name__ == '__main__':
    logger = Logger("test.log", "TEST")
    logger.setRotation()
    logger.log("Test Debug")
    logger.logobj.error("Test Error")
