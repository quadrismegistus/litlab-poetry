litlab-poetry
=============

Code used in the [Literary Lab](http://litlab.stanford.edu)'s [Trans-historical Poetry Project](http://litlab.stanford.edu/?page_id=13). We presented this project at DH2014 in Lausanne. Abstract is [here](http://dharchive.org/paper/DH2014/Paper-788.xml), but this is relatively uninformative. Slides are [here](https://docs.google.com/presentation/d/1KyCi4s6P1fE4D3SlzlZPnXgPjwZvyv_Vt-aU3tlb24I/edit?usp=sharing).

# Quick start
## Run Open Mary
Open Mary (http://mary.dfki.de/) is an open-source Text-to-Speech software, which is used here in order to syllabify and provide stress annotations for words not included in the CMU Pronunciation Dictionary. In order for a poem to be parsed, you'll need first to run OpenMary as a server. This is how to do that:

```bash
cd marytts-5.0/bin/
./marytts-server.sh
```

## Parse a poem
