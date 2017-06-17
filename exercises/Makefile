UNAME := $(shell uname)


basics.model.complexity: code/basics.model.complexity.py
	@python $<
basics.model.selection: code/basics.model.selection.py
	@python $<
applications.peptide.binding: code/applications.peptide.binding.py
	@python $<
download.data:
	@wget -b http://graal.ift.ulaval.ca/adrouin/ml-tutorial/cdiff_clindamycin.kover -P ./data/antibiotics
download.verify:
ifeq ($(UNAME), Linux)
	@md5sum -c <<<"f0b7bcf295f370da3440198ada98e104 ./data/antibiotics/cdiff_clindamycin.kover"
endif
ifeq ($(UNAME), Darwin)
	$(eval KOVERDSMD5 := $(shell md5 -q ./data/antibiotics/cdiff_clindamycin.kover))
ifeq ($(KOVERDSMD5), "﻿f0b7bcf295f370da3440198ada98e104")
	@echo "Kover dataset: OK."
else
	@echo "Kover dataset: FAILED."
endif
endif
