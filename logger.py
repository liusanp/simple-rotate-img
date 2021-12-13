#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import datetime
import logging.handlers
from config import logRoot, logName


if not os.path.exists(logRoot):
    os.makedirs(logRoot)


def get_logger():
    pid = str(os.getpid())
    p_logger = logging.getLogger(logName)
    p_logger.setLevel(logging.INFO)

    rf_handler = logging.handlers.TimedRotatingFileHandler('{}/all.log'.format(logRoot), when='midnight',
                                                           backupCount=7, atTime=datetime.time(0, 0, 0, 0),
                                                           encoding='utf-8')
    rf_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    f_handler = logging.FileHandler('{}/error.log'.format(logRoot), encoding='utf-8')
    f_handler.setLevel(logging.ERROR)
    f_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s"))

    s_handler = logging.StreamHandler()
    s_handler.setLevel(logging.INFO)
    s_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    p_logger.addHandler(rf_handler)
    p_logger.addHandler(f_handler)
    p_logger.addHandler(s_handler)

    return p_logger


# 处理快照进程日志
logger = get_logger()
