# coding: utf-8
'''
:Author: Stefan Wallner
:Description: Logger class, Created on Thursday 04 05 2023
'''

from __future__ import absolute_import, print_function, division, annotations

import logging
import sys
import threading
import contextlib


class Filters():

	class NameOutFilter(logging.Filter):
		'''
		Rejects log-records where the name is staring with the given name
		'''

		def __init__(self, name): # pylint: disable=super-init-not-called
			self._filter_out_name = name

		def filter(self, record):
			allow = not record.name.startswith(
			    self._filter_out_name)
			return allow

	class LevelOutFilter(logging.Filter):
		'''
		Rejects log-records where the level is equal or above the given level
		'''

		def __init__(self, level): # pylint: disable=super-init-not-called
			self._filter_out_level = level

		def filter(self, record):
			return record.levelno < self._filter_out_level

class Logger:
	"""
	Logger class provides a customizable logging utility for the sphysics application.
	It supports features such as colored output, indentation management, progress tracking,
	and dynamic formatting for console logs. The class is designed to handle multi-threaded
	logging and offers various logging levels (debug, info, warning, error, critical) with
	optional color-coded output.

	Attributes:
		sphysicsRoot (logging.Logger): Root logger for the sphysics application.
		cliHandlerStdout (logging.StreamHandler): Handler for stdout logs.
		cliHandlerStderr (logging.StreamHandler): Handler for stderr logs.

	Methods:
		setCLILevel: Sets the logging level for CLI output.
		printTimeStamp: Toggles the inclusion of timestamps in CLI logs.
		incrementIndent: Increases the indentation level for the current thread.
		decrementIndent: Decreases the indentation level for the current thread.
		indented: Context manager for temporary indentation.
		setLevel: Sets the logging level for the logger instance.
		debug: Logs a message with DEBUG level.
		info: Logs a message with INFO level.
		emph: Logs a message with INFO level and blue color.
		success: Logs a message with INFO level and green color.
		warning: Logs a message with WARNING level.
		error: Logs a message with ERROR level.
		critical: Logs a message with CRITICAL level and exits the program.
		raiseException: Logs a critical error and raises an exception.
		_log: Internal method to log messages with optional color and indentation.
		printLine: Logs a line of repeated characters.
		initProgress: Initializes a progress bar for tracking tasks.
		updateProgress: Updates the progress bar with completed tasks.
		finishProgress: Marks the progress bar as complete.
		printProgress: Prints the current progress bar state.
		printFinishProgress: Prints a completion message for the progress bar.
	"""

	_formatString = "{color}[{levelname}: %(name)-{namewidth}.{namewidth}s{time}] %(message)-{messagewidth}s{colorend}"  # color, namewidth, messagewidth, time, colorend
	_cliShowTime = False
	_cliShowColor = False
	_cliNameWidth = 20
	_cliLineWidth = None
	_cliLevelnameShort = True

	sphysicsRoot = None
	cliHandlerStdout = logging.StreamHandler(sys.stdout)
	cliHandlerStderr = logging.StreamHandler(sys.stderr)

	_currentLevel = {'MainThread': 0}

	_colorcodes = {
	    'white': '\x1b[0m',
	    'red': '\x1b[31m',
	    'green': '\x1b[32m',
	    'blue': '\x1b[34m',
	    'magenta': '\x1b[35m',
	}

	_loggerLock = threading.Lock()
	_loggerPrefixLock = threading.Lock()
	_loggerIndentLock = threading.Lock()

	@classmethod
	def _genFormatter(cls, color, nameWidth, messageWidth, time,
	                  levelnameShort):
		'''Generates log formatter with current settings

		:param color: If true, color place-holder is inserted
		:param nameWidth: Width of scope-name label
		:param messageWidth: Width of message. If None, no limit on the message width
		:param time: If True, time is added
		:param levelnameShort: Use short level name instead of long one
		:return: Formatter with the given settings
		:rtype: logging.Formatter
		'''

		headerWidth = 5
		headerWidth += 3 if levelnameShort else 7
		headerWidth += nameWidth
		headerWidth += 20 if time else 0
		if messageWidth:
			messageWidth -= headerWidth

		color = "%(ccode)s" if color else ""
		nameWidth = int(nameWidth)
		messageWidth = ".{0}".format(
		    int(messageWidth)) if messageWidth else ""
		time = ", %(asctime)-.16s" if time else ""
		colorend = "\x1b[0m" if color else ""
		levelname = "%(levelname)-7.7s" if not levelnameShort else "%(levelname)-3.3s"

		return logging.Formatter(
		    cls._formatString.format(color=color,
		                              colorend=colorend,
		                              namewidth=nameWidth,
		                              messagewidth=messageWidth,
		                              time=time,
		                              levelname=levelname))

	@classmethod
	def _setCLIFormatterSettings(cls):
		'''
		Generate the new CLI formatter with the current settings
		'''
		cls.cliHandlerStdout.setFormatter(
		    cls._genFormatter(color=cls._cliShowColor,
		                      nameWidth=cls._cliNameWidth,
		                      messageWidth=cls._cliLineWidth,
		                      time=cls._cliShowTime,
		                      levelnameShort=cls._cliLevelnameShort))
		cls.cliHandlerStderr.setFormatter(
		    cls._genFormatter(color=cls._cliShowColor,
		                      nameWidth=cls._cliNameWidth,
		                      messageWidth=cls._cliLineWidth,
		                      time=cls._cliShowTime,
		                      levelnameShort=cls._cliLevelnameShort))

	@classmethod
	def _initialize(cls):
		'''Initializes file logging, sets logger level to DEBUG, sets the output handler to INFO level, and the error handler to WARNING level.

    	:param cls: The class object.
		'''
		cls._cliShowColor = True

		cls.sphysicsRoot = logging.getLogger('sphysics')
		cls.sphysicsRoot.setLevel(logging.DEBUG)

		Logger.cliHandlerStdout.setLevel(logging.INFO)
		Logger.cliHandlerStdout.addFilter(
			Filters.LevelOutFilter(logging.WARNING))
		Logger.cliHandlerStderr.setLevel(logging.WARNING)
		cls.sphysicsRoot.addHandler(Logger.cliHandlerStdout)
		cls.sphysicsRoot.addHandler(Logger.cliHandlerStderr)
		Logger._setCLIFormatterSettings()

	@classmethod
	def setCLILevel(cls, level):
		'''
		Sets the output level on the command line

		:param level: Number of output levels
		'''
		with cls._loggerLock:
			cls.cliHandlerStdout.setLevel(level)

	@classmethod
	def printTimeStamp(cls, do=True): # pylint: disable=invalid-name
		'''Determines if timestamps are included in the logs

		:param do: If True, timestamps are included, if false, timestamps are removed. Defaults to True.
		'''
		with cls._loggerLock:
			cls._cliShowTime = bool(do)
			cls._setCLIFormatterSettings()

	def __init__(self, name: str):
		if name != 'sphysics':
			name = 'sphysics.'+name
		self._logger = logging.getLogger(name)
		self._progressTotalTasks = 0
		self._progressFinalTasks = 0

	def incrementIndent(self):
		'''Increases indentation level for current thread
		'''
		with self._loggerIndentLock:
			tname = threading.current_thread().name
			Logger._currentLevel[tname] = Logger._currentLevel.setdefault(
				tname, Logger._currentLevel['MainThread']) + 1

	def decrementIndent(self):
		'''Decreases indentation level for current thread
		'''
		with self._loggerIndentLock:
			tname = threading.current_thread().name
			Logger._currentLevel[tname] = Logger._currentLevel.setdefault(
				tname, Logger._currentLevel['MainThread']) - 1

	@contextlib.contextmanager
	def indented(self):
		'''Increases indentation level temporarily
		'''
		try:
			self.incrementIndent()
			yield self
		finally:
			self.decrementIndent()

	def _getIndentLevel(self):
		'''Retrieves the current indentation level for the current thread

		:return: Current indentation level
		'''
		tname = threading.current_thread().name
		with self._loggerIndentLock:
			indent = Logger._currentLevel.setdefault(
				tname, Logger._currentLevel['MainThread'])
		return indent


	def setLevel(self, level: int):
		'''Sets the logging level to `level`

		:param level: logging level
		:type level: int
		'''
		self._logger.setLevel(level)


	def debug(self, msg, *args, **kwargs):
		'''Logs a message at the 'DEBUG' level

		:param msg: Log message
		'''
		self._log(logging.DEBUG, msg, None, *args, **kwargs)

	def info(self, msg, *args, **kwargs):
		'''Logs a message at the 'INFO' level

		:param msg: Log message
		'''
		self._log(logging.INFO, msg, None, *args, **kwargs)

	def emph(self, msg, *args, **kwargs):
		'''Logs an emphasized (blue-coloured) message at the 'INFO' level

		:param msg: Log message
		'''
		self._log(logging.INFO, msg, 'blue', *args, **kwargs)

	def success(self, msg, *args, **kwargs):
		'''Logs a successful (green-coloured) message at the 'INFO' level

		:param msg: Log message
		'''
		self._log(logging.INFO, msg, 'green', *args, **kwargs)

	def warning(self, msg, *args, **kwargs):
		'''Logs a warning message at the 'Warning' level

		:param msg: Log message
		'''
		self._log(logging.WARNING, msg, None, *args, **kwargs)

	def error(self, msg, *args, **kwargs):
		'''Logs an error message at the 'Error' level

		:param msg: Log message
		'''
		self._log(logging.ERROR, msg, None, *args, **kwargs)

	def critical(self, msg, *args, **kwargs):
		'''Logs a critical message at the 'Critical' level

		:param msg: Log message
		'''
		self._log(logging.CRITICAL, msg, None, *args, **kwargs)
		sys.exit(666)

	def raiseException(self, e_type, msg, *args, **kwargs):
		'''
		Write exception to logger and raise the exception

		:param e_type: Exception type
		'''
		parsed_msg = self._log(logging.CRITICAL, msg, None, *args, **kwargs)
		raise e_type(parsed_msg)

	def _log(self, level, msg, color_out=None, *args, **kwargs): # pylint: disable=keyword-arg-before-vararg
		extras = {}
		if color_out is None:
			if level == logging.DEBUG:
				extras = {'ccode': Logger._colorcodes['white']}
			elif level == logging.INFO:
				extras = {'ccode': Logger._colorcodes['white']}
			elif level == logging.WARNING:
				extras = {'ccode': Logger._colorcodes['magenta']}
			elif level == logging.ERROR:
				extras = {'ccode': Logger._colorcodes['red']}
			elif level == logging.CRITICAL:
				extras = {'ccode': Logger._colorcodes['red']}
		else:
			extras = {'ccode': Logger._colorcodes[color_out]}

		for line in msg.split('\n'):
			line = '	' * self._getIndentLevel() + line
			self._logger.log(level, line, *args, extra=extras, **kwargs)

		return msg

	def printLine(self, c="=", length=80):
		'''Prints a line of characters of specified length

		:param c: character to be printed, defaults to "="
		:param length: Length of the line, defaults to 80
		'''
		self.info(c * length)

	def initProgress(self, total_tasks):
		'''
		Initializes logger for printing a progress bar

		:param total_tasks: Total number of tasks
		'''
		self._progressTotalTasks = total_tasks
		self._progressFinalTasks = 0
		self.printProgress(finishedTasks=self._progressFinalTasks,
		                   totalTasks=self._progressTotalTasks)

	def updateProgress(self, increment=1):
		'''
		Updates the process bar by a specified increment

		:param increment: number of tasks completed, defaults to 1
		'''
		self._progressFinalTasks += increment
		self.printProgress(finishedTasks=self._progressFinalTasks,
		                   totalTasks=self._progressTotalTasks)

	def finishProgress(self):
		'''
		Finishes the progress bar
		'''
		self._progressTotalTasks = None
		self.printFinishProgress()

	def printProgress(self, finishedTasks=None, totalTasks=None):
		'''
		Prints the ongoing progress. If finishedTasks is not given, it just prints one dot for each call.

		:param finishedTasks: The number of tasks completed. Defaults to None.
		:param totalTasks: The total number of tasks. Defaults to None.
		'''
		leading_ws = 10
		progressbarWidth = 20

		if finishedTasks is not None:
			if totalTasks is None:
				raise AttributeError(
				    "printProgress called with finished_tasks but without total_tasks"
				)
			if totalTasks > 0:
				n = int(
				    float(finishedTasks) / totalTasks * progressbarWidth)
			elif totalTasks == 0 and finishedTasks == 0:
				n = progressbarWidth
			else:
				raise AttributeError("printProgressbar called with " +
				                     str(finishedTasks) + " and with " +
				                     str(totalTasks) + " total_tasks!")
			msg = "\r" + " " * leading_ws
			msg += "[" + '=' * n + ' ' * (progressbarWidth - n) + ']'
			msg += " {0:3.0f}%".format(
			    float(finishedTasks) / totalTasks * 100)
		else:
			msg = '.'
		sys.stdout.write(msg)
		sys.stdout.flush()

	def printFinishProgress(self):
		'''Prints a message when the progress is finished
		'''
		msg = " done\n"
		sys.stdout.write(msg)
		sys.stdout.flush()


Logger._initialize() # pylint: disable=protected-access
