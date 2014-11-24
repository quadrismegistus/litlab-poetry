litlab-poetry
=============

Code used in the [Literary Lab](http://litlab.stanford.edu)'s [Trans-historical Poetry Project](http://litlab.stanford.edu/?page_id=13), involving myself ([Ryan Heuser](http://twitter.com/quadrismegistus)), [Mark Algee-Hewitt](https://twitter.com/mark_a_h), Maria Kraxenberger, J.D. Porter, Jonny Sensenbaugh, and Justin Tackett. We presented the project at DH2014 in Lausanne. The abstract is [here](http://dharchive.org/paper/DH2014/Paper-788.xml), but a better source of information is our slideshow (with notes) [here](https://docs.google.com/presentation/d/1KyCi4s6P1fE4D3SlzlZPnXgPjwZvyv_Vt-aU3tlb24I/edit?usp=sharing). We plan to publish the results of our 2+ year project as a Lab pamphlet sometime in 2015. Feel free to use this code for any purpose whatever, but please provide attribution back to (for now) this page or to the Literary Lab website.

The goal in the project is to develop software capable of annotating the following four features of poetic form:

1. Stanzaic scheme (Syllable scheme / beat scheme) [**Complete**]:
  * An example scheme is: _10_ (Invariable) or _8-6_ (Alternating) or _10-10-10-10-10-6_ (Complex)
  * Invariable schemes (e.g. Inv_10 = the poem is generally always in lines of 10 syllables in length, e.g. blank verse, sonnets, heroic couplets)
  * Alternating schemes (e.g. _Alt_8_6_ = the poem alternates between lines of 8 and 6 syllables in length. Most common in ballads)
  * Complex schemes (basically, everything more complex than the above two. Includes odes, free verse, etc)

2. Metrical scheme [**Complete**]
Produce a scansion of each of the poem's lines, and then decide if the poem's meter is predominantly:
    1. Iambic (Binary foot, head final)
    2. Trochaic (Binary foot, head initial)
    3. Anapestic (Ternary foot, head final)
    4. Dactylic (Ternary foot, head initial)

3. Rhyme scheme [**Ongoing**]
  * Determine the rhyme scheme of the poem.

4. Synthetic form [**Ongoing**]
  * From the above (#1-3) elements, decide if poem is, e.g.:
    * Heroic couplets = ([1] Inv_10, [2] iambic, [3] aa)
    * Blank verse = ([1] Inv_10, [2] iambic, [3] unrhymed)
    * etc.


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

This loads each line in the ```lined``` dictionary into another dictionary, ```prosodic```, also keyed by Line ID, but this time each leading to a Prosodic "Text" object. A convenient way to look at the parses is using the ```parse_str()``` method:

```python
for lineid,lineObj in poem.prosodic.items():
	print lineid,lineObj.parse_str()
```
This should return:
```
(1, 1) WHO|will|GO|drive.with*|FER|gus|NOW
(2, 1) and|PIERCE|the|DEEP|wood's*|WOV|en|SHADE
(3, 1) and|DANCE|up|ON|the|LE|vel|SHORE
(4, 1) YOUNG|man*||LIFT|up.your|RU|sset|BROW
(5, 1) and|LIFT|your|TEN|der|EYE|lids*||MAID
(6, 1) and|BROOD|on|HOPES|and|FEAR|no*|MORE
(7, 2) and|NO|more|TURN|as|IDE|and|BROOD
(8, 2) up|ON|love's*|BI|tter|MY|st.ery
(9, 2) for|FER|gus|RULES|the|BRA|zen|CARS
(10, 2) and|RULES|the|SHA|dows|OF*|the|WOOD
(11, 2) AND*|the.white*|BREAST|of.the|DIM|sea*
(12, 2) and|ALL|di|SH|evelled|WA|nd.ering|STARS
```
The * indicate a metrical violation occurred in that position. Also, ```parse_str()``` has two important keyword arguments: ```text``` and ```viols```, either of which can be ```True``` or ```False```. For instance, running the same command in this way:
```python
for lineid,lineObj in poem.prosodic.items():
	print lineid,lineObj.parse_str(text=False, viols=False)
```
Returns:
```
(1, 1) s|w|s|ww|s|w|s
(2, 1) w|s|w|s|w|s|w|s
(3, 1) w|s|w|s|w|s|w|s
(4, 1) s|w||s|ww|s|w|s
(5, 1) w|s|w|s|w|s|w||s
(6, 1) w|s|w|s|w|s|w|s
(7, 2) w|s|w|s|w|s|w|s
(8, 2) w|s|w|s|w|s|ww
(9, 2) w|s|w|s|w|s|w|s
(10, 2) w|s|w|s|w|s|w|s
(11, 2) s|ww|s|ww|s|w
(12, 2) w|s|w|s|w|s|ww|s
```
Which is a more abstract representation of the metrical output.


### Get statistics from poem's parsing

All of these are contained in another dictionary each poem object has, its ```statd``` -- but this becomes available _only_ after the ```poem.parse()``` command is executed. Here are the available statistics on a poem:

```python
for featname,featval in sorted(poem.statd.items()):
	print featname,'\t',featval
```
Should return:
```
beat_scheme 	(4,)
beat_scheme_diff 	2
beat_scheme_length 	1
beat_scheme_repr 	Inv_4
beat_scheme_type 	Invariable
meter_ambiguity 	3.0
meter_constraint_TOTAL 	0.120879120879
meter_constraint_footmin-no-s 	0.0
meter_constraint_strength_s=>-u 	0.0
meter_constraint_strength_w=>-p 	0.010989010989
meter_constraint_stress_s=>-u 	0.021978021978
meter_constraint_stress_w=>-p 	0.0879120879121
meter_length_avg_line 	8.08333333333
meter_length_avg_parse 	6.92857142857
meter_mpos_s 	0.505494505495
meter_mpos_w 	0.428571428571
meter_mpos_ww 	0.0659340659341
meter_perc_lines_ending_s 	0.833333333333
meter_perc_lines_ending_w 	0.166666666667
meter_perc_lines_fourthpos_s 	0.833333333333
meter_perc_lines_fourthpos_w 	0.166666666667
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

The relevant keys corresponding to our goals #1 and #2 (above, at the top of this readme) are:

1. ```syll_scheme_repr``` - In this case, Inv_8, meaning lines are invariably eight lines long. [In scheme, not in actuality.]
2. ```meter_type_scheme``` - In this case, iambic, meaning feet are generally binary with their stress at the end (head final).