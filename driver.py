from markov import Markov

def genetator(path):
	file_ = open(path)
	gen = Markov(file_)
	return gen.generate
