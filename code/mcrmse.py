import numpy

# Function definition to compute Mean Columnwise Root Mean Square Error between two series
def compute_error(prediction, gt):
	return numpy.mean(numpy.sqrt(numpy.mean((prediction - gt)**2, axis=0)))
