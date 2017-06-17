UNAME := $(shell uname)


basics.model.complexity: code/basics.model.complexity.py
	@python $<
basics.model.selection: code/basics.model.selection.py
	@python $<
applications.peptide.binding: code/applications.peptide.binding.py
	@python $<
download.data:
	@wget -bN http://graal.ift.ulaval.ca/adrouin/ml-tutorial/cdiff_clindamycin.kover -P ./data/antibiotics
download.verify:
ifeq ($(UNAME), Linux)
	@md5sum ./data/antibiotics/cdiff_clindamycin.kover
endif
ifeq ($(UNAME), Darwin)
	md5 -r ./data/antibiotics/cdiff_clindamycin.kover
endif
