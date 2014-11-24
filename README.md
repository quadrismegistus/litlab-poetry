litlab-poetry
=============

Code used in the Literary Lab's Trans-historical Poetry Project.

# Quick start
## Run Open Mary
Open Mary (http://mary.dfki.de/) is an open-source Text-to-Speech software, which is used here in order to syllabify and provide stress annotations for words not included in the CMU Pronunciation Dictionary. In order for a poem to be parsed, you'll need first to run OpenMary as a server. This is how to do that:

```bash
cd marytts-5.0/bin/
./marytts-server.sh
```

## Parse a poem
