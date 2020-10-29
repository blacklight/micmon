import logging
import sys

logging.basicConfig(level=logging.INFO,
                    stream=sys.stdout,
                    format='[%(asctime)s] %(name)s|%(levelname)-8s|%(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

