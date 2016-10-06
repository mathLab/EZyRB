class Space(object):
	def __init__(self):
		raise NotImplemented

	def __call__(self, value):
		raise NotImplemented

	def save(self, filename):
		raise NotImplemented

	def load(self, filename):
		raise NotImplemented
