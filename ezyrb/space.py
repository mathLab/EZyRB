"""
Module with the base class for a generic space.
"""

class Space(object):
	"""
	Abstract class.
	"""
	def __init__(self):
		raise NotImplemented

	def __call__(self, value):
		"""
		Abstract method to approximate the value of a generic point.

		Not implemented, it has to be implemented in subclasses.
		"""
		raise NotImplemented

	def save(self, filename):
		"""
		Abstract method to save the space to a specific file.

		Not implemented, it has to be implemented in subclasses.
		"""
		raise NotImplemented

	def load(self, filename):
		"""
		Abstract method to load the space from a specific file.

		Not implemented, it has to be implemented in subclasses.
		"""
		raise NotImplemented
