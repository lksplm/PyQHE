import argparse
import numpy as np
from scipy.special import factorial
from itertools import product
from math import sqrt, factorial
from pyqhe.basis import BasisFermi
from pyqhe.hamiltonian import OperatorLinCy, OperatorQuadCy, OperatorQuadDeltaCy
from pyqhe.eigensystem import Eigensystem
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("N", type=int, help="Number of particles per spin component")#
parser.add_argument("m", type=int, help="orbital cutoff per spin component")
parser.add_argument("-a", "--alpha", type=int, help="increase output verbosity", default=100)
args = parser.parse_args()


print(args.N)
print(args.m)
print(args.N)