litlab-poetry
=============

Code used in the [Literary Lab](http://litlab.stanford.edu)'s [Trans-historical Poetry Project](http://litlab.stanford.edu/?page_id=13), involving myself ([Ryan Heuser](http://twitter.com/quadrismegistus)), [Mark Algee-Hewitt](https://twitter.com/mark_a_h), Maria Kraxenberger, J.D. Porter, Jonny Sensenbaugh, and Justin Tackett. We presented the project at DH2014 in Lausanne. The abstract is [here](http://dharchive.org/paper/DH2014/Paper-788.xml), but a better source of information is our slideshow (with notes) [here](https://docs.google.com/presentation/d/1KyCi4s6P1fE4D3SlzlZPnXgPjwZvyv_Vt-aU3tlb24I/edit?usp=sharing). We plan to publish the results of our 2+ year project as a Lab pamphlet sometime in 2015.


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

poem = pypoesy.PoemTXT("""Who will go drive with Fergus now,
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
```
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
As you can see, the Line ID is actually a tuple of integers: (Line #, Stanza #).

### Parse a poem metrically

Metrical parsing is done via [Prosodic](https://github.com/quadrismegistus/prosodic), developed by Arto Antilla, Josh Falk, and Ryan Heuser. To parse a poem:

```python
poem.parse()
```

### Get statistics from poem's parsing

There are essentially two domains of information that we are currently able to provide about a poem: its stanzaic structure (what we call its syllable scheme); and its metrical patterns. All of these are contained in another dictionary each poem object has, its ```statd``` -- but this becomes available _only_ after the ```poem.parse()``` command is executed. Here are the available statistics on a poem:

```python
for k,v in sorted(p.statd.items()):
	print k,'\t',v
```
Should return:
```
beat_scheme 	(4,)
beat_scheme_diff 	4
beat_scheme_length 	1
beat_scheme_repr 	Inv_4
beat_scheme_type 	Invariable
meter_ambiguity 	3.08333333333
meter_constraint_TOTAL 	0.137931034483
meter_constraint_footmin-no-s 	0.0
meter_constraint_strength_s=>-u 	0.0
meter_constraint_strength_w=>-p 	0.0229885057471
meter_constraint_stress_s=>-u 	0.0114942528736
meter_constraint_stress_w=>-p 	0.103448275862
meter_length_avg_line 	8.08333333333
meter_length_avg_parse 	6.92857142857
meter_mpos_s 	0.505747126437
meter_mpos_w 	0.379310344828
meter_mpos_ww 	0.114942528736
meter_perc_lines_ending_s 	0.833333333333
meter_perc_lines_ending_w 	0.166666666667
meter_perc_lines_fourthpos_s 	0.666666666667
meter_perc_lines_fourthpos_w 	0.333333333333
meter_perc_lines_starting_s 	0.25
meter_perc_lines_starting_w 	0.75
meter_type_foot 	binary
meter_type_head 	final
meter_type_scheme 	iambic
num_lines 	12
num_lines_group 	0011-20
syll_scheme 	(8,)
syll_scheme_diff 	1
syll_scheme_length 	1
syll_scheme_repr 	Inv_8
syll_scheme_type 	Invariable
```