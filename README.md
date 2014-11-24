litlab-poetry
=============

Code used in the [Literary Lab](http://litlab.stanford.edu)'s [Trans-historical Poetry Project](http://litlab.stanford.edu/?page_id=13). We presented this project at DH2014 in Lausanne. Abstract is [here](http://dharchive.org/paper/DH2014/Paper-788.xml), but this is relatively uninformative. Slides (with notes) are [here](https://docs.google.com/presentation/d/1KyCi4s6P1fE4D3SlzlZPnXgPjwZvyv_Vt-aU3tlb24I/edit?usp=sharing). We plan to publish the results of these experiments into poetic form in the summer of 2015.

## Quick start
### Run Open Mary
Open Mary (http://mary.dfki.de/) is an open-source Text-to-Speech software, which is used here in order to syllabify and provide stress annotations for words not included in the CMU Pronunciation Dictionary. In order for a poem to be parsed, you'll need first to run OpenMary as a server. To do that, run this in your terminal:

```bash
cd marytts-5.0/bin/
./marytts-server.sh
```

### Parse a poem
The main module here is _pypoesy.py_ and the main class within it, _PoemTXT_. PoemTXT actually runs off of the class Poem, but this was built explicitly for the purpose of running poems stored in a custom data format for Chadwyck's XML files.

```python
import pypoesy as pp
poem = pp.Poem("""Who will go drive with Fergus now,
And pierce the deep wood's woven shade,
And dance upon the level shore?
Young man, lift up your russet brow,
And lift your tender eyelids, maid,
And brood on hopes and fear no more.

And no more turn aside and brood
Upon love's bitter mystery;
For Fergus rules the brazen cars,
And rules the shadows of the wood,
And the white breast of the dim sea
And all dishevelled wandering stars.""")
```
