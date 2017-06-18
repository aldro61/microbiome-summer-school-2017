UNAME := $(shell uname)


basics.model.complexity: code/basics.model.complexity.py
	@python $<
basics.model.selection: code/basics.model.selection.py
	@python $<
applications.peptide.binding: code/applications.peptide.binding.py
	@python $<
install.dependencies:
	sudo pip install cython numpy scipy matplotlib scikit-learn seaborn h5py --upgrade && cd code/GSkernel_source && python build.py build_ext -b gs_kernel/ && cd .. && ln -s GSkernel_source/gs_kernel
applications.antibiotics.data:
	@wget -bN http://graal.ift.ulaval.ca/adrouin/kover-example-data.zip -P ./data/antibiotics && unzip ./data/antibiotics/kover-example-data.zip
