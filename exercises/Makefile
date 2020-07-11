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
	@wget -N https://github.com/aldro61/kover-tutorial-data/releases/download/1.0.0/kover-example-data.zip -P ./data/antibiotics && unzip ./data/antibiotics/kover-example-data.zip
applications.antibiotics.svm: code/applications.antibiotics.svm.py
	@python $<
