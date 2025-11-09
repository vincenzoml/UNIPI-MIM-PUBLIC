VOXLOGICA=~/bin/VoxLogicA/VoxLogicA

.PHONY: default run clean output2

default: output output2

all.imgql: example??.imgql exampleGlobal??.imgql
	cat $^ > $@

output: all.imgql
	$(VOXLOGICA) $<

output2: all2.imgql
	$(VOXLOGICA) $<

all2.imgql: exampleAdvanced??.imgql
	cat $^ > $@

clean: 
	rm -rf all.imgql all2.imgql output 
