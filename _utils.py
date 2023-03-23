#!/usr/bin/python

import sys
import time
import psutil
import logging
import threading
import tracemalloc

from typing import Union
from functools import wraps

# DECORATORS
def perf_monitor(func):
	""" Measure performance of a function """
	@wraps(func)
	def wrapper(*args, **kwargs):
		strt_time = time.perf_counter()
		cpu_percent_prev = psutil.cpu_percent(interval=0.05, percpu=False)
		tracemalloc.start()
		try:
			return func(*args, **kwargs)
		except Exception as e:
			logging.exception(f"Exception in {func.__name__}: {e}", exc_info=True, stack_info=True)
		finally:
			current, peak = tracemalloc.get_traced_memory()
			tracemalloc.stop()
			cpu_percent = psutil.cpu_percent(interval=None, percpu=False)
			cpu_percnt = cpu_percent - cpu_percent_prev
			end_time = time.perf_counter()
			duration = end_time - strt_time
			msj = f"{func.__name__}\t\tUsed {abs(cpu_percnt):>5.1f} % CPU: {hm_time(duration)}\t Mem: [avr:{hm_sz(current):>8}, max:{hm_sz(peak):>8}]\t({func.__doc__})"
			logging.info(msj)
	return wrapper

def show_running_message_decorator(func):
	@wraps(func)
	def wrapper(*args, **kwargs):
		message = f" {func.__name__} running"

		def progress_indicator():
			sys.stdout.write(message)
			while not progress_indicator.stop:
				for pattern in "|/-o+\\":
					sys.stdout.write(f"\r{message} {pattern}")
					sys.stdout.flush()
					time.sleep(0.1)
			sys.stdout.write(f"\r{message} Done!\n")
			sys.stdout.flush()

		progress_indicator.stop = False
		progress_thread = threading.Thread(target=progress_indicator)
		progress_thread.start()

		try:
			result = func(*args, **kwargs)
		finally:
			progress_indicator.stop = True
			progress_thread.join()

		return result
	return wrapper

'''
# Example usage
@show_running_message_decorator
def some_long_running_function():
	time.sleep(5)

some_long_running_function()
'''

def measure_cpu_time(func):
	def wrapper(*args, **kwargs):
		start_time = time.time()
		cpu_percent = psutil.cpu_percent(interval=None, percpu=True)
		result = func(*args, **kwargs)
		elapsed_time = time.time() - start_time
		cpu_percent = [p - c for p, c in zip(psutil.cpu_percent(interval=None, percpu=True), cpu_percent)]
		print(f"Function {func.__name__} used {sum(cpu_percent)/len(cpu_percent)}% CPU over {elapsed_time:.2f} seconds")
		return result
	return wrapper


def logit(logfile='out.log', de_bug=False):
	def logging_decorator(func):
		@wraps(func)
		def wrapper(*args, **kwargs):
			result = func(*args, **kwargs)
			with open(logfile, 'a') as f:
				if len(kwargs) > 0:
					f.write(f"\n{func.__name__}{args} {kwargs} = {result}\n")
				else:
					f.write(f"\n{func.__name__}{args} = {result}\n")
			if de_bug:
				if len(kwargs) > 0:
					print(f"{func.__name__}{args} {kwargs} = {result}")
				else:
					print(f"{func.__name__}{args} = {result}")
			return result
		return wrapper
	return logging_decorator


def handle_exception(func):
	"""Decorator to handle exceptions."""
	@wraps(func)
	def wrapper(*args, **kwargs):
		try:
			return func(*args, **kwargs)
		except Exception as e:
			print(f"Exception in {func.__name__}: {e}")
			logging.exception(f"Exception in {func.__name__}: {e}",exc_info=True, stack_info=True)
#			sys.exit(1)
		except TypeError :
			print(f"{func.__name__} wrong data types")
		except IOError:
			print("Could not write to file.")
		except :
			print("Someting Else?")
		else:
			print("No Exceptions")
		finally:
			logging.error("Error: ", exc_info=True)
			logging.error("uncaught exception: %s", traceback.format_exc())
	return wrapper


def measure_cpu_utilization(func):
	"""Measure CPU utilization, number of cores used, and their capacity."""
	@wraps(func)
	def wrapper(*args, **kwargs):
		cpu_count = psutil.cpu_count(logical=True)
		strt_time = time.monotonic()
		cpu_prcnt = psutil.cpu_percent(interval=0.1, percpu=True)
		result = func(*args, **kwargs)
		end_time = time.monotonic()
		cpu_percnt = sum(cpu_prcnt) / cpu_count
		return result, cpu_percnt, cpu_prcnt
	return wrapper

def log_exceptions(func):
	"""Log exceptions that occur within a function."""
	@wraps(func)
	def wrapper(*args, **kwargs):
		try:
			return func(*args, **kwargs)
		except Exception as e:
			print(f"Exception in {func.__name__}: {e}")
			logging.exception(f"Exception in {func.__name__}: {e}",exc_info=True, stack_info=True)
	return wrapper

def measure_execution_time(func):
	"""Measure the execution time of a function."""
	@wraps(func)
	def wrapper(*args, **kwargs):
		strt_time = time.perf_counter()
		result = func(*args, **kwargs)
		end_time = time.perf_counter()
		duration = end_time - strt_time
		print(f"{func.__name__}: Execution time: {duration:.5f} sec")
		return result
	return wrapper

def measure_memory_usage(func):
	"""Measure the memory usage of a function."""
	@wraps(func)
	def wrapper(*args, **kwargs):
		tracemalloc.start()
		result = func(*args, **kwargs)
		current, peak = tracemalloc.get_traced_memory()
		print(f"{func.__name__}: Mem usage: {current / 10**6:.6f} MB (avg), {peak / 10**6:.6f} MB (peak)")
		tracemalloc.stop()
		return result
	return wrapper

def performance_check(func):
	"""Measure performance of a function"""
	@log_exceptions
	@measure_execution_time
	@measure_memory_usage
	@measure_cpu_utilization
	@wraps(func)
	def wrapper(*args, **kwargs):
		return func(*args, **kwargs)
	return wrapper

def temperature ():
	sensors = psutil.sensors_temperatures()
	for name, entries in sensors.items():
		print(f"{name}:")
		for entry in entries:
			print(f"  {entry.label}: {entry.current}°C")

def perf_monitor_temp(func):
	""" Measure performance of a function """
	@wraps(func)
	def wrapper(*args, **kwargs):
		strt_time           = time.perf_counter()
		cpu_percent_prev    = psutil.cpu_percent(interval=0.05, percpu=False)
		tracemalloc.start()
		try:
			return func(*args, **kwargs)
		except Exception as e:
			logging.exception(f"Exception in {func.__name__}: {e}",exc_info=True, stack_info=True)
		finally:
			current, peak   = tracemalloc.get_traced_memory()
			tracemalloc.stop()
			cpu_percent     = psutil.cpu_percent(interval=None, percpu=False)
			cpu_percnt      = cpu_percent - cpu_percent_prev

			# New code to measure CPU temperature
			cpu_temp = psutil.sensors_temperatures().get('coretemp')[0].current
			print(f"CPU temperature: {cpu_temp}°C")

			end_time        = time.perf_counter()
			duration        = end_time - strt_time
			msj = f"{func.__name__}\t\tUsed {abs(cpu_percnt):>5.1f} % CPU: {hm_time(duration)}\t Mem: [avr:{hm_sz(current):>8}, max:{hm_sz(peak):>8}]\t({func.__doc__})"
			logging.info(msj)
	return wrapper

##>>============-------------------<  End  >------------------==============<<##

#  CLASES
# XXX: https://shallowsky.com/blog/programming/python-tee.html

class Tee:
	''' implement the Linux Tee function '''

	def __init__(self, *targets):
		self.targets = targets

	def __del__(self):
		for target in self.targets:
			if target not in (sys.stdout, sys.stderr):
				target.close()

	def write(self, obj):
		for target in self.targets:
			try:
				target.write(obj)
				target.flush()
			except Exception:
				pass

	def flush(self):
		pass

class RunningAverage:
	''' Compute the running averaga of a value '''

	def __init__(self):
		self.n = 0
		self.avg = 0

	def update(self, x):
		self.avg = (self.avg * self.n + x) / (self.n + 1)
		self.n += 1

	def get_avg(self):
		return self.avg

	def reset(self):
		self.n = 0
		self.avg = 0
##>>============-------------------<  End  >------------------==============<<##
## Functions

def hm_sz(numb: Union[str, int, float], type: str = "B") -> str:
	'''convert file size to human readable format'''
	numb = float(numb)
	try:
		if numb < 1024.0:
			return f"{numb} {type}"
		for unit in ['','K','M','G','T','P','E']:
			if numb < 1024.0:
				return f"{numb:.2f} {unit}{type}"
			numb /= 1024.0
		return f"{numb:.2f} {unit}{type}"
	except Exception as e:
		logging.exception(f"Error {e}", exc_info=True, stack_info=True)
		print (e)
#		traceback.print_exc()
##==============-------------------   End   -------------------==============##

def hm_time(timez: float) -> str:
	'''Print time as years, months, weeks, days, hours, min, sec'''
	units = {'year': 31536000,
			 'month': 2592000,
			 'week': 604800,
			 'day': 86400,
			 'hour': 3600,
			 'min': 60,
			 'sec': 1,
			}
	if timez < 0:
		return "Error negative"
	elif timez == 0 :
		return "Zero"
	elif timez < 0.001:
		return f"{timez * 1000:.3f} ms"
	elif timez < 60:
		return f"{timez:>5.3f} sec{'s' if timez > 1 else ''}"
	else:
		frmt = []
		for unit, seconds_per_unit in units.items() :
			value = timez // seconds_per_unit
			if value != 0:
				frmt.append(f"{int(value)} {unit}{'s' if value > 1 else ''}")
			timez %= seconds_per_unit
		return ", ".join(frmt[:-1]) + " and " + frmt[-1] if len(frmt) > 1 else frmt[0] if len(frmt) == 1 else "0 sec"


##>>============-------------------<  End  >------------------==============<<##

def file_size(path):
	# Return file/dir size (MB)
	mb = 1 << 20  # bytes to MiB (1024 ** 2)
	path = Path(path)
	if path.is_file():
		return path.stat().st_size / mb
	elif path.is_dir():
		return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()) / mb
	else:
		return 0.0
##>>============-------------------<  End  >------------------==============<<##
