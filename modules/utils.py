import logging
from time import strftime
from time import localtime

def timeNow():
    return strftime('%Y-%M-%d %H:%m:%2ds', localtime())


def getLogger(className='', level=logging.INFO):
    logger = logging.getLogger(f'module/{className}')
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(level)
    consoleFormatter = logging.Formatter(f"%(filename)s/{className}[%(levelname)s]:\n %(message)s")
    consoleHandler.setFormatter(consoleFormatter)
    logger.addHandler(consoleHandler)
    return logger

# Global logger
Log = getLogger()