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

### Loading a poem
The main module here is _pypoesy.py_ and the main class within it, _PoemTXT_. PoemTXT actually runs off of the class Poem, but this was built explicitly for the purpose of running poems stored in a custom data format for Chadwyck's XML files, so I've added _PoemTXT_ in order to allow the loading of poems as strings, with a double line-break indicating a stanzaic break:

```python
import pypoesy

poem = pypoesy.Poem("""Who will go drive with Fergus now,
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

This loads the poem into the _lined_ property of the object, which is a dictionary keyed by the Line ID and whose value is a string representation of the line. When looping over the lines in a poem, make sure to sort as you go:

```python
for lineid,line in sorted(poem.lined.items()):
	print lineid,line
```
This should return:
```bash
(1, 1) Who will go drive with Fergus now,
(2, 1) And pierce the deep wood's woven shade,
(3, 1) And dance upon the level shore?
(4, 1) Young man, lift up your russet brow,
(5, 1) And lift your tender eyelids, maid,
(6, 1) And brood on hopes and fear no more.
(7, 2) And no more turn aside and brood
(8, 2) Upon love's bitter mystery;
(9, 2) For Fergus rules the brazen cars,
(10, 2) And rules the shadows of the wood,
(11, 2) And the white breast of the dim sea
(12, 2) And all dishevelled wandering stars.
```

### Print 