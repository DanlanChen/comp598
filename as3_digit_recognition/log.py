import sys
from datetime import datetime

def log(string):
	sys.stderr.write('[' + str(datetime.now()) + '] ' + str(string) + '\n')

