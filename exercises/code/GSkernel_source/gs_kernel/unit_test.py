import numpy as np

from GSkernel_fast import GS_gram_matrix_fast
from GSkernel import GS_gram_matrix

def load_matrix(file_name):
	f = open(file_name)
	lines = f.readlines()
	f.close()
	
	M = []
	for l in lines:
		M.append([float(x) for x in l.split()])
	
	return np.array(M)
	
def test():
	amino_acid_property_file = 'amino_acids_matrix/AA.blosum50.dat'
	sigma_position = 1.0
	sigma_amino_acid = 1.0
	substring_length = 3
	
	f = open('examples/data/Zhou2010_bradykinin.dat')
	bradykinin = [l.split()[0] for l in f.readlines()]
	f.close()
	
	f = open('examples/data/Zhou2010_cationic.dat')
	cationic = [l.split()[0] for l in f.readlines()]
	f.close()
	
	print "Testing normalized symetric matrix"
	K1 = GS_gram_matrix_fast(X=bradykinin,
							Y=bradykinin,
							amino_acid_property_file=amino_acid_property_file,
							sigma_position=sigma_position,
							sigma_amino_acid=sigma_amino_acid,
							substring_length=substring_length,
							normalize_matrix=True)
	
	K2 = GS_gram_matrix(	X=bradykinin,
							Y=bradykinin,
							amino_acid_property_file=amino_acid_property_file,
							sigma_position=sigma_position,
							sigma_amino_acid=sigma_amino_acid,
							substring_length=substring_length,
							normalize_matrix=True)
	
	assert(np.allclose(K1,K2))
	
	print "Testing un-normalized symetric matrix"
	K1 = GS_gram_matrix_fast(X=bradykinin,
							Y=bradykinin,
							amino_acid_property_file=amino_acid_property_file,
							sigma_position=sigma_position,
							sigma_amino_acid=sigma_amino_acid,
							substring_length=substring_length,
							normalize_matrix=False)
	
	K2 = GS_gram_matrix(	X=bradykinin,
							Y=bradykinin,
							amino_acid_property_file=amino_acid_property_file,
							sigma_position=sigma_position,
							sigma_amino_acid=sigma_amino_acid,
							substring_length=substring_length,
							normalize_matrix=False)
	
	assert(np.allclose(K1,K2))
	
	print "Testing normalized non-symetric matrix"
	K1 = GS_gram_matrix_fast(X=bradykinin,
							Y=cationic,
							amino_acid_property_file=amino_acid_property_file,
							sigma_position=sigma_position,
							sigma_amino_acid=sigma_amino_acid,
							substring_length=substring_length,
							normalize_matrix=True)
	
	K2 = GS_gram_matrix(	X=bradykinin,
							Y=cationic,
							amino_acid_property_file=amino_acid_property_file,
							sigma_position=sigma_position,
							sigma_amino_acid=sigma_amino_acid,
							substring_length=substring_length,
							normalize_matrix=True)
	
	assert(np.allclose(K1,K2))
	
	print "Testing un-normalized non-symetric matrix"
	K1 = GS_gram_matrix_fast(X=bradykinin,
							Y=cationic,
							amino_acid_property_file=amino_acid_property_file,
							sigma_position=sigma_position,
							sigma_amino_acid=sigma_amino_acid,
							substring_length=substring_length,
							normalize_matrix=False)
	
	K2 = GS_gram_matrix(	X=bradykinin,
							Y=cationic,
							amino_acid_property_file=amino_acid_property_file,
							sigma_position=sigma_position,
							sigma_amino_acid=sigma_amino_acid,
							substring_length=substring_length,
							normalize_matrix=False)
	
	assert(np.allclose(K1,K2))




    