"""

"""


import prosodic as p,os,pytxt
p.config['print_to_screen']=0
roots={'poetry':'../corpora/corppoetry', 'prose':'../corpora/corpprose'}
old=[]

MAKE_DATA=True
if MAKE_DATA:
	for rootgenre,rootdir in roots.items():
		for fn in os.listdir(rootdir):
			if not fn.endswith('.txt'): continue
			lang=fn.split('.')[0]
			t = p.Text(os.path.join(rootdir,fn),lang=lang,limWord=5000)

			for li,line in enumerate(t.lines()):
				linedx={}
				linedx['fn']=fn
				linedx['genre_root']=rootgenre
				linedx['lang']=lang
				linedx['name']='.'.join(fn.split('.')[1:-1])
				linedx['line_num']=li+1
				linedx['line']=repr(line)
				linedx['line_num_syll']=len(line.syllables())
				linedx['line_num_word']=len(line.words())

				numsyll=0
				fp=[]
				stresses = [word.stress for word in line.words()]
				if '?' in stresses: continue
				for syll in line.syllables():
					numsyll+=1
					fp+=syll.feature_pairs
				fpd=pytxt.toks2freq(fp)
				for k in fpd:
					linedx[k]=fpd[k]/float(numsyll)
				

				print linedx				
				old+=[linedx]

	pytxt.write2('genre_results_by_line.txt', old)
	exit()


ld=pytxt.tsv2ld('genre_results.txt')
ld=[d for d in ld if d['lang']=='fi']
import rpyd2
r=rpyd2.RpyD2(ld)
for col in r.q().cols:
	r.aov(col+"~genre_root", plot=True,plotOpts={'title':None, 'text':'fn','pdf':True})