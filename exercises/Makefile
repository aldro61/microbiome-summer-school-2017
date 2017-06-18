UNAME := $(shell uname)


basics.model.complexity: code/basics.model.complexity.py
	@python $<
basics.model.selection: code/basics.model.selection.py
	@python $<
applications.peptide.binding: code/applications.peptide.binding.py
	@python $<
download.data:
	@wget -bN http://graal.ift.ulaval.ca/adrouin/kover-example.zip -P ./data/antibiotics
download.verify:
ifeq ($(UNAME), Linux)
	@md5sum ./data/antibiotics/KmerMatrix.tsv
	@md5sum ./data/antibiotics/metadata.tsv
endif
ifeq ($(UNAME), Darwin)
	md5 -r ./data/antibiotics/KmerMatrix.tsv
	md5 -r ./data/antibiotics/metadata.tsv
endif
install.gskernel:
	@cd code/GSkernel_source; \
	python build.py build_ext -b gs_kernel/; \
	cd ..; \
	ln -s GSkernel_source/gs_kernel
install.dependencies:
	sudo pip install cython numpy scipy matplotlib scikit-learn seaborn h5py --upgrade && cd code/GSkernel_source && python build.py build_ext -b gs_kernel/ && cd .. && ln -s GSkernel_source/gs_kernel
