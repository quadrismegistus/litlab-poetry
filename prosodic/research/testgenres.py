import prosodic as p,os,pytxt
p.config['print_to_screen']=0
roots={'poetry':'corpora/corppoetry', 'prose':'corpora/corpprose'}
old=[]

MAKE_DATA=False
if MAKE_DATA:
	for rootgenre,rootdir in roots.items():
		for fn in os.listdir(rootdir):
			if not fn.endswith('.txt'): continue
			lang=fn.split('.')[0]
			t = p.Text(os.path.join(rootdir,fn),lang=lang,limWord=5000)
			print t
			numsyll=0
			fp=[]
			for syll in t.syllables():
				numsyll+=1
				fp+=syll.feature_pairs
			fpd=pytxt.toks2freq(fp)
			for k in fpd:
				fpd[k]=fpd[k]/float(numsyll)
			

			fpd['fn']=fn
			fpd['genre_root']=rootgenre
			fpd['lang']=lang
			fpd['name']='.'.join(fn.split('.')[1:-1])
			old+=[fpd]

	pytxt.write2('genre_results.txt', old)
	exit()


ld=pytxt.tsv2ld('genre_results.txt')
ld=[d for d in ld if d['lang']=='fi']
import rpyd2
r=rpyd2.RpyD2(ld)
for col in r.q().cols:
	r.aov(col+"~genre_root", plot=True,plotOpts={'title':None, 'text':'fn','pdf':True})