## encoding=utf-8
from __future__ import division
import sys,os,codecs
import pytxt,pystats,cPickle


class Author(object):
	
	def __init__(self,poetid):
		self.id=poetid
		self.dbfn=os.path.join(poetfolder,str(poetid)+'.tml')
		self.genn=False
		self.data={}
		self.concordance={}
		self.numWords=0
	
	def __str__(self):
		if not self.genn: self.gen()
		
		return str(self.lname)
		
	def key(self):
		return str(self.lname)
	
	
	
	def numCites(self,type='raw'):
		if not self.genn: self.gen()
		try:
			return self.data['count']['type'][type]
		except KeyError:
			return 0
	
	def loadConcordance(self):
		#
		concfn=os.path.join(folder_concordance_poet,str(self.id)+'.tsv')
		"""
		self.concordance=pytxt.readDict(concfn)
		if self.concordance==None:
			self.concordance={}
			return None

		if len(self.concordance):
			self.numWords=sum(self.concordance.values())
			return self.concordance
		"""
		
		if not self.genn: self.gen()
		for poem in self.poems():
			
			#if not len(poem.concordance): poem.loadConcordance()
			poem.gen()
			poem.loadConcordance()
			for word,count in poem.concordance.items():
				try:
					self.concordance[word]+=count
				except KeyError:
					self.concordance[word]=count
		self.numWords=sum(self.concordance.values())
		# from operator import itemgetter
		# 		for w,c in sorted(self.concordance.items(),reverse=True,key=itemgetter(1))[0:25]:
		# 			print "\t".join(str(x) for x in [w,c])
		#if len(self.concordance):
		#	pytxt.writeDict(concfn,self.concordance)
	
	def count(self,word):
		if not len(self.concordance): self.loadConcordance()
		try:
			return self.concordance[word]
		except KeyError:
			return 0
				
	def freq(self,word):
		if not len(self.concordance): self.loadConcordance()
		try:
			return self.concordance[word]/self.numWords
		except KeyError:
			return 0
	
	def stats(self,poetobj=None,fieldwords={},ntopwords=0):
		r={}
		

		try:
			r['numcites']=poetobj.numCites(type='percent_max')/0.5
		except:
			r['numcites']=None
			
		r['tf_punc_commas']=self.freq(',')
		r['tf_punc_exclmark']=self.freq('!')
		r['tf_punc_period']=self.freq('.')
		r['tf_punc_semicolon']=self.freq(';')
		r['tf_punc_colon']=self.freq(':')
		r['tf_punc_question']=self.freq('?')
		r['tf_punc_hyphen']=self.freq('-')
		r['tf_punc_quote']=self.freq('"')
		r['tf_punc_apostrophe']=self.freq("'")

		r['tf_punc']=sum([r[x] for x in r.keys() if x.startswith("tf_punc_")])

		wordlens=[]
		for k,v in self.concordance.items():
			for n in range(v):
				wordlens+=[len(k)]
		if not len(wordlens): return {}
		wordlen,wordstd=pystats.mean_stdev(wordlens)
		#print wordlen

		r['wordlen_avg']=wordlen
		r['tf_i']=self.freq('i')
		r['typtok']=len(self.concordance.keys())/sum(self.concordance.values())
		#r['numpoems']=len(self.poems())
		r['tf_you']=self.freq('you')
		r['tf_heart']=self.freq('heart')
		r['tf_love']=self.freq('love')
		r['tf_god']=self.freq('god')
		r['tf_she']=self.freq('she')
		r['tf_lips']=self.freq('lips')
		r['tf_eyes']=self.freq('eyes')
		r['tf_thou']=self.freq('thou')
		r['shannon_wordlens']=pystats.shannon(wordlens)
		r['shannon_numwordtoks']=pystats.shannon(self.concordance.values())
		r['numWords']=self.numWords

		#r['wordsperline']=sum(len(x.lines()) for x in self.poems())/self.numWords
		for field,words in fieldwords.items():
			field='fieldtf_'+"".join(field.split(".")[1:])
			r[field]=0.0
			for word in words:
				r[field]+=self.freq(word)
		
		if ntopwords:
			from operator import itemgetter
			n=0
			for word,score in sorted(self.concordance.items(),key=itemgetter(1),reverse=True):
				n+=1
				if n>ntopwords: break
				r['tf_'+word]=self.freq(word)
		
		try:
			r['numcites']=poetobj.dob
		except:
			r['numcites']=None

		return r
	
	def gen(self):
		print ">> gen: "+self.dbfn
		f=open(self.dbfn)
		t=f.read()
		f.close()
		
		self.data=pytxt.extractTagsAsDict(t)
		#print self.data
		#exit()
		
		for k,v in self.data.items():
			if k=="poem": continue
			if not hasattr(self,k):
				setattr(self,k,v)
		
		if len(self.data):
			self.genn=True
	
	def poems(self,gen=False):
		if not self.genn: self.gen()
		
		try:
			for poemid,poemobj in self.data['poem']['id'].items():
				if type(poemobj) in [str,unicode]:
					poem=Poem(poemid)
					if gen: poem.gen()
					self.data['poem']['id'][poemid]=poem
					poemobj=poem
				else:
					if gen: poemobj.gen()
	
			return [x for x in self.data['poem']['id'].values() if isinstance(x,Poem)]
		except KeyError:
			return []
		
	def poem(self,poemid):
		return Poem(poemid)
		
		if not self.genn: self.gen()
		
		if type(self.data['poem']['id'][poemid]) in [str,unicode]:
			self.data['poem']['id'][poemid]=Poem(poemid)
			self.data['poem']['id'][poemid].gen()
		return self.data['poem']['id'][poemid]
	

class Poem(Author):
	def __init__(self,poemid=None):
		"""
		Initialize a poem object with its ID as input
		"""
		self.id=poemid
		self.dbfn=os.path.join(poemfolder,str(poemid)+'.tml') if poemid else None
		self.genn=False
		self.data={}
		self.concordance={}
		self.numWords=0
		self.numLines=0

	@property
	def stanza_length(self):
		"""
		Returns invariable stanza length as an integer.
		If variable stanza lengths, returns None.
		"""
		if hasattr(self,'_stanza_length'): return self._stanza_length
		
		import bs4
		dom=bs4.BeautifulSoup(self.tml)
		for stanzatype in ['stanza','versepara']:
			stanzas=list(self.dom.find_all('div',{'type':stanzatype}))
			if len(stanzas): break
		else:
			self._stanza_length=None
			return None
		
		stanza_lens=[len(list(stanza('div',{'type':'line'}))) for stanza in stanzas]
		
		if not stanza_lens or len(set(stanza_lens))>1:
			self._stanza_length=None # return None if variable stanza lengths
		else:
			self._stanza_length=stanza_lens[0]
		return self._stanza_length

	@property
	def meterd(self):
		"""
		Return dictionary with metrical annotations.
		"""
		datad={}
		all_mstrs=[]
		ws_ends=[]
		ws_starts=[]
		ws_fourths=[]
		
		## COLLECT STATS ON metrically weak/strong in various positions
		viold={}
		for i,line in sorted(self.prosodic.items()):
			bp=line.bestParses()
			if not bp: continue
			mstrs=[]
			pure_parse=''
			for parse in bp:
				for mpos in parse.positions:
					for ck,cv in mpos.constraintScores.items():
						ck=ck.name
						if not ck in viold: viold[ck]=[]
						viold[ck]+=[0 if not cv else 1]
					mstr=''.join([mpos.meterVal for n in range(len(mpos.slots))])
					pure_parse+=mstr
					mstrs+=[mstr]
			line_mstr='|'.join(mstrs)
			line_parse="||".join(str(p) for p in bp)
			ws_starts += [pure_parse[0]]
			ws_ends += [pure_parse[-1]]
			ws_fourths += [pure_parse[3:4]]
			all_mstrs+=mstrs

		mstr_freqs=pytxt.toks2freq(all_mstrs,tfy=True)
		for k,v in mstr_freqs.items(): datad['mpos_'+k]=v
		for k,v in pytxt.toks2freq(ws_ends,tfy=True).items(): datad['perc_lines_ending_'+k]=v
		for k,v in pytxt.toks2freq(ws_starts,tfy=True).items(): datad['perc_lines_starting_'+k]=v
		for k,v in pytxt.toks2freq(ws_fourths,tfy=True).items(): datad['perc_lines_fourthpos_'+k]=v

		
		## DECIDE WHETHER TERNARY / BINARY FOOT
		d=datad
		d['type_foot']='ternary' if d.get('mpos_ww',0)>0.175 else 'binary'

		## DECIDE WHETHER INITIAL / FINAL-HEADED
		if d['type_foot']=='ternary':
			d['type_head']='initial' if d.get('perc_lines_fourthpos_s',0)>0.5 else 'final'
		else:
			d['type_head']='initial' if d.get('perc_lines_fourthpos_s',0)<0.5 else 'final'
		
		## PUT 2 TOGETHER TO DECIDE anapestic / dactylic / trochaic / iambic
		x=(d['type_foot'],d['type_head'])
		if x==('ternary','final'):
			d['type_scheme']='anapestic'
		elif x==('ternary','initial'):
			d['type_scheme']='dactylic'
		elif x==('binary','initial'):
			d['type_scheme']='trochaic'
		else:
			d['type_scheme']='iambic'


		### METRICAL AMBIGUITY
		ambig = []
		avg_linelength=[]
		avg_parselength=[]
		for i,line in sorted(self.prosodic.items()):
			ap=line.allParses()
			line_numparses=[]
			line_parselen=0
			if not ap: continue
			for parselist in ap:
				numparses=len(parselist)
				parselen=len(parselist[0].str_meter())
				avg_parselength+=[parselen]
				line_parselen+=parselen
				line_numparses+=[numparses]
			
			import operator
			avg_linelength+=[line_parselen]
			ambigx=reduce(operator.mul, line_numparses, 1)
			ambig+=[ambigx]
		d['ambiguity']=pystats.mean(ambig)
		d['length_avg_line']=pystats.mean(avg_linelength)
		d['length_avg_parse']=pystats.mean(avg_parselength)


		## TOTAL METRICAL VIOLATIONS
		sumviol=0
		for ck,cv in viold.items():
			avg=pystats.mean(cv)
			d['constraint_'+ck.replace('.','_')]=avg
			sumviol+=avg

		d['constraint_TOTAL']=sumviol



		return datad


	def __str__(self):
		"""
		Return a string version of poem: its ID
		"""
		return self.id
	
	@property
	def tml(self):
		if not hasattr(self,'_tml'):
			import codecs
			print ">> opening",self.dbfn
			f=codecs.open(self.dbfn,encoding='utf-8')
			tml=f.read()
			f.close()
			## text-cleaning operations
			tml=tml.replace('&indent;','&nbsp;&nbsp;&nbsp;&nbsp;')
			tml=tml.replace('&wblank;','')
			tml=tml.replace('&lblank;','')
			tml=tml.replace('&rblank;','')
			tml=tml.replace(' & ',' and ')
			tml=tml.replace(u"‘",'"')
			tml=tml.replace(u"’",'"')
			tml=tml.replace('versepara','stanza')
			###
			self._tml=tml
		return self._tml
	
	@property
	def dom(self):
		if not hasattr(self,'_dom'):
			import bs4
			dom=bs4.BeautifulSoup(self.tml)
			poem=dom.find('body')
			## cleaning operations
			for tagtype in [('edit',{}), ('div',{'type':'note'})]:
				for badtag in poem.find_all(tagtype[0],tagtype[1]):
					badtag.extract()
			####
			## get rid of redundant "first line" div
			for firstline in dom('div',{'type':'firstline'}):
				new_contents=[]
				for tag in firstline.parent.contents:
					if tag!=firstline:
						new_contents+=[tag]
					else:
						for subtag in firstline.contents:
							new_contents+=[subtag]
				
				firstline.parent.contents = new_contents
				
			
			## standardize and number stanzas
			stanzanum=0
			for i,stanza in enumerate(dom('div',{'type':'stanza'})):
				if True in [hasattr(child,'get') and child.get('type','')=='line' for child in stanza.contents]:
					stanzanum+=1
				
				stanza['n']=stanzanum
			
			self._dom=dom
		return self._dom
	
	def limit(self,N):
		""" Limit number of lines to first N """	
		newd={}
		for i,(k,v) in enumerate(sorted(self.lined.items())):
			if i>=N: break
			newd[k]=v
		self.lined=newd

	
	def gen(self,force=False):
		global poemshelf
		if self.genn: return
		
		if poemshelf and not force:
			_fnfn=os.path.join(poemshelf,self.id)
			if os.path.exists(_fnfn):
				print ">> gen {0} from poemshelf...".format(self.id)
				for k,v in cPickle.load(open(_fnfn)).items():
					self.__dict__[k]=v
				print "\t>> done."
				return
		
		## runs only if not from shelf
		print ">> gen {0} from TML...".format(self.id)
		try:
			self.title=self.data['title']=self.dom.find('title').find('main').text
		except AttributeError:
			self.title=self.data['title']=self.dom.find('title').text
		self.lined=self.extract_lines()
		self.numLines=len(self.lined)
		#self.lines=[line for index,line in sorted(self.lined.items())]	
		print "\t>> done."
		self.genn=True
		return
	
	@property
	def d(self):
		if not hasattr(self,'_d'):
			## begin / metadata
			dx={'id':self.id}
			for tag in self.dom.find('head'):
				if not hasattr(tag,'find_all'): continue
				for tag2 in tag.find_all():
					dx[tag.name+'_'+tag2.name]=tag2.text
			
			## Make integers out of dates
			for k in ['author_dob','author_dod']:
				if not k in dx: continue
			
				try:
					dx[k]=int(dx[k])
					dx[k+'_valid']=True
				except ValueError:
					dx[k+'_valid']=False
			
			## set
			self._d=dx
		
		## return
		return self._d
	
	
	def save(self,not2save=['_dom']):
		global poemshelf
		_fnfn=os.path.join(poemshelf,self.id)
		saved=dict((k,v) for k,v in self.__dict__.items() if not k in not2save)
		
		if '_prosodic' in saved:
			for i,textobj in saved['_prosodic'].items():
				#textobj.dict.refresh()
				textobj.dict=None
		
		cPickle.dump(saved, open(_fnfn,'wb'))
		print ">> saved:",_fnfn
	
	
	@property
	def indices(self):
		return sorted(self.lined.keys())

	
	@property
	def meterd(self):
		datad={}
		all_mstrs=[]
		ws_ends=[]
		ws_starts=[]
		ws_fourths=[]
		
		viold={}
		for i,line in sorted(self.prosodic.items()):
			bp=line.bestParses()
			if not bp: continue
			mstrs=[]
			pure_parse=''
			for parse in bp:
				for mpos in parse.positions:
					for ck,cv in mpos.constraintScores.items():
						ck=ck.name
						if not ck in viold: viold[ck]=[]
						viold[ck]+=[0 if not cv else 1]

					mstr=''.join([mpos.meterVal for n in range(len(mpos.slots))])
					pure_parse+=mstr
					mstrs+=[mstr]
			line_mstr='|'.join(mstrs)
			line_parse="||".join(str(p) for p in bp)
			ws_starts += [pure_parse[0]]
			ws_ends += [pure_parse[-1]]
			ws_fourths += [pure_parse[3:4]]
			
			all_mstrs+=mstrs

		mstr_freqs=pytxt.toks2freq(all_mstrs,tfy=True)
		for k,v in mstr_freqs.items(): datad['mpos_'+k]=v

		for k,v in pytxt.toks2freq(ws_ends,tfy=True).items(): datad['perc_lines_ending_'+k]=v
		for k,v in pytxt.toks2freq(ws_starts,tfy=True).items(): datad['perc_lines_starting_'+k]=v
		for k,v in pytxt.toks2freq(ws_fourths,tfy=True).items(): datad['perc_lines_fourthpos_'+k]=v

		d=datad
		#d['type_foot']='ternary' if d.get('mpos_ww',0)>0.15 and d.get('mpos_w',0)<.35 else 'binary'
		d['type_foot']='ternary' if d.get('mpos_ww',0)>0.175 else 'binary'

		if d['type_foot']=='ternary':
			d['type_head']='initial' if d.get('perc_lines_fourthpos_s',0)>0.5 else 'final'
		else:
			d['type_head']='initial' if d.get('perc_lines_fourthpos_s',0)<0.5 else 'final'
		

		x=(d['type_foot'],d['type_head'])
		if x==('ternary','final'):
			d['type_scheme']='anapestic'
		elif x==('ternary','initial'):
			d['type_scheme']='dactylic'
		elif x==('binary','initial'):
			d['type_scheme']='trochaic'
		else:
			d['type_scheme']='iambic'


		### AMBIGUITY
		ambig = []
		avg_linelength=[]
		avg_parselength=[]
		for i,line in sorted(self.prosodic.items()):
			ap=line.allParses()
			line_numparses=[]
			line_parselen=0
			if not ap: continue
			for parselist in ap:
				numparses=len(parselist)
				parselen=len(parselist[0].str_meter())
				avg_parselength+=[parselen]
				line_parselen+=parselen
				line_numparses+=[numparses]
			
			import operator
			avg_linelength+=[line_parselen]
			ambigx=reduce(operator.mul, line_numparses, 1)
			ambig+=[ambigx]
		d['ambiguity']=pystats.mean(ambig)
		d['length_avg_line']=pystats.mean(avg_linelength)
		d['length_avg_parse']=pystats.mean(avg_parselength)


		## VIOLATIONS
		sumviol=0
		for ck,cv in viold.items():
			avg=pystats.mean(cv)
			d['constraint_'+ck.replace('.','_')]=avg
			sumviol+=avg

		d['constraint_TOTAL']=sumviol



		return datad

	@property
	def meter(self):
		return self.meterd['type_scheme']

	@property
	def parsed(self):
		if hasattr(self,'_parsed'): return self._parsed
		meterd=self.meterd

		"""
		Tie-breaker logic:
		anapestic --> maximize ww, start with w
		trochaic --> minimize ww, start with s
		dactylic --> maximize ww, start with s
		iambic --> minimize ww, start with w
		"""
		self._parsed=parsed={}
		def sort_ties(ties, meterd):
			ww_factor=-1 if meterd['type_foot']=='ternary' else 1
			wstart_factor=1 if meterd['type_head']=='initial' else -1
			def _sort_tuple(P):
				num_ww=sum([int(mpos.mstr=='ww') for mpos in P.positions])
				zero_means_starts_with_w=int(P.positions[0].mstr[0]!='w')
				return (num_ww,zero_means_starts_with_w)

			ties.sort(key=lambda P: (ww_factor*_sort_tuple(P)[0], wstart_factor*_sort_tuple(P)[1]))

		for i,line in sorted(self.prosodic.items()):
			ap=line.allParses()
			if not ap: continue
			parsed[i]=lineparses=[]
			for parselist in ap:	# first level of list is punctuation breaks
				## here is where we decide among parses based on which maximizes metrical scheme
				## only if there are ties
				parselist.sort(key=lambda P: P.totalScore)
				lowestScore=parselist[0].totalScore
				ties=[P for P in parselist if P.totalScore==lowestScore]
				if len(ties)>1: sort_ties(ties,meterd)
				lineparses+=[ties[0]]
		
		return parsed





	def parse(self,lim=None):
		for _i,(li,line) in enumerate(self.prosodic.items()):
			if lim and _i>=lim: break
			line.parse()
	
	@property
	def openmary(self):
		if not hasattr(self,'_openmary'):
			self._openmary=pd={}
			numlines=len(self.lined)
			for _i,(i,line) in enumerate(sorted(self.lined.items())):
				#print u'>> openmary:\t{0}\t{1}\t{2}'.format(_i+1,numlines,line)
				pd[i]=openmary(line)
				#pd[i]=standardize(line)
		return self._openmary

	@property
	def lengroup(self):
		numlines=self.numLines
		if numlines<=10:
			return '0000-10'
		elif numlines<=20:
			return '0011-20'
		elif numlines<=100:
			return '0021-100'
		elif numlines<=1000:
			return '0100-1000'
		else:
			return '1000+'

	@property
	def schemed(self):
		if hasattr(self,'_schemed'): return self._schemed
		self._schemed=dx={}
		dx['scheme']=self.scheme
		dx['scheme_type']=self.schemetype
		dx['scheme_repr']=self.scheme_repr
		dx['scheme_length']=len(self.scheme)
		dx['scheme_diff']=self._scheme_diff
		return dx

	def get_schemed(self,beat=True):
		scheme,sdiff=self.get_scheme(beat=beat,return_diff=True)
		dx={}
		dx['scheme']=scheme
		dx['scheme_type']=self.schemetype(scheme)
		dx['scheme_repr']=self.scheme_repr(dx['scheme_type'], dx['scheme'])
		dx['scheme_length']=len(scheme)
		dx['scheme_diff']=sdiff
		return dx



	@property
	def statd(self):
		dx={}
		## Scheme
		for x,y in [('beat',True), ('syll',False)]:
			sd=self.get_schemed(beat=y)
			for sk,sv in sd.iteritems():
				dx[x+'_'+sk]=sv

		## Length
		dx['num_lines']=self.numLines
		dx['num_lines_group']=self.lengroup

		## Meter
		for k,v in self.meterd.items(): dx['meter_'+k]=v


		return dx




	def schemetype(self,scheme):
		if len(scheme)==1: return 'Invariable'
		if len(scheme)==2: return 'Alternating'
		return 'Complex'

	def scheme_repr(self,schemetype,scheme):
		if schemetype=='Complex': return 'Complex'
		prefix='Alt_' if schemetype=='Alternating' else 'Inv_'
		return prefix+'_'.join(str(sx) for sx in scheme)


	def get_scheme(self,beat=True,return_diff=False):
		stanza_length=self.stanza_length
		if beat:
			lengths=[v for k,v in sorted(self.linelengths_bybeat.items())]
		else:
			lengths=[v for k,v in sorted(self.linelengths.items())]
		num_lines=len(lengths)
		min_length,max_length=min(lengths),max(lengths)
		abs_diff_in_lengths=abs(min_length-max_length)

		if beat:
			isVariable=True if abs_diff_in_lengths>2 else False
		else:
			isVariable=True if abs_diff_in_lengths>4 else False


		min_seq_length=1 # if not isVariable else 2
		try_lim=10
		max_seq_length=stanza_length if stanza_length else 12
		

		def measure_diff(l1,l2):
			min_l=min([len(l1),len(l2)])
			l1=l1[:min_l]
			l2=l2[:min_l]
			"""print len(l1),len(l2)
			print '  '.join(str(x) for x in l1)
			print '  '.join(str(x) for x in l2)
			print '  '.join(str(abs(x1-x2)) for x1,x2 in zip(l1,l2))"""
			diff=0
			for x1,x2 in zip(l1,l2):
				diff+=abs(x1-x2)
			return diff

		combo2diff={}
		best_combo=None
		best_diff=None

		best_combos=[]
		best_lim=100
		poem_length=self.numLines
		for seq_length in range(min_seq_length,int(max_seq_length)+1):
			if seq_length>poem_length: break
			if stanza_length and stanza_length % seq_length: continue
			if poem_length and poem_length % seq_length: continue
			if seq_length>try_lim: break
			#if seq_length!=2: continue
			num_reps=num_lines/seq_length
			#print "best combo so far:",best_combo,best_diff
			print "Sequence Length: {0} x {1} = {2}".format(seq_length,num_reps,num_lines)

			average_length_per_pos=dict((s_i,[]) for s_i in range(seq_length))
			for l_i,l_x in enumerate(lengths):
				average_length_per_pos[l_i % seq_length] += [l_x]
			for k,v in average_length_per_pos.items():
				median=pystats.median(v) if len(v)>1 else v[0]
				average_length_per_pos[k]=int(median)

			#ALL_possibilities = [[rx for rx in range(min_length,max_length+1)] for x_i,x in enumerate(range(seq_length))]
			SOME_possibilities = [[rx for rx in range(average_length_per_pos[x_i]-1,average_length_per_pos[x_i]+2)] for x_i,x in enumerate(range(seq_length))]
			combo_possibilities = list(pystats.product(*SOME_possibilities))
			#print "\t# possibilities =",len(combo_possibilities)
			for combo in combo_possibilities:
				if len(combo)>1 and len(set(combo))==1: continue
				model_lengths=[]
				while len(model_lengths)<=len(lengths):
					for cx in combo: model_lengths+=[cx]
				model_lengths=model_lengths[:len(lengths)]

				diff_in_lengths=abs(len(lengths) - len(model_lengths))
				diff=measure_diff(lengths, model_lengths)
				#if isVariable and len(combo)==1:
				#	diff+=1000
				#diff+=(diff_in_lengths*2)
				if not beat:
					diff+=sum([5 if seq_x%2 else 0 for seq_x in combo])
				#else:
				#for n in combo[1:]:
					#diff+=2
				diff=diff


				if len(best_combos)<best_lim or diff<max([_d for _c,_d in best_combos]):
					best_combos+=[(combo,diff)]
					best_combos=sorted(best_combos,key=lambda _lt: _lt[1])[:best_lim]

				if best_diff==None or diff<best_diff:
					best_diff=diff
					best_combo=combo

				elif best_combo and diff<=best_diff:
					if len(combo)<len(best_combo):
						best_combo=combo
						best_diff=diff
					elif pystats.mean(combo)>pystats.mean(best_combo):
						best_diff=diff
						best_combo=combo

		
		self._scheme=best_combo
		self._scheme_diff=best_diff
		for bc,bd in sorted(best_combos,key=lambda _lt: -_lt[1])[-5:]:
			print bc,bd


		print "SCHEME:",best_combo,best_diff
		if return_diff:
			return best_combo,best_diff

		return best_combo

	@property
	def scheme(self):
		#if hasattr(self,'_scheme'): return self._scheme
		return self.get_scheme(beat=True)
	
	@property
	def prosodic(self):
		if not hasattr(self,'_prosodic'):
			import prosodic as p
			p.config['print_to_screen']=0
			self._prosodic=pd={}
			numlines=len(self.lined)
			for _i,(i,line) in enumerate(sorted(self.openmary.items())):
				#print u'>> prosodic:\t{0}\t{1}\t{2}'.format(_i+1,numlines,line)
				pd[i]=p.Text(line)
		return self._prosodic
	
	
	def count(self,word):
		if not len(self.concordance): self.loadConcordance()
		try:
			return self.concordance[word]
		except KeyError:
			return 0

	def freq(self,word):
		if not len(self.concordance): self.loadConcordance()
		try:
			return self.concordance[word]/self.numWords
		except KeyError:
			return 0
	
	def loadConcordance(self):
		if not self.genn: self.gen()
		for line in self.lines:
			for word in pytxt.tokenize2(line,punc=False):
				word=word.strip().lower()
				if not word: continue
				try:
					self.concordance[word]+=1
				except KeyError:
					self.concordance[word]=1
		self.numWords=sum(self.concordance.values())
	
	def txt0(self):
		if not self.genn: self.gen(force=True)
		stanzanumnow=0
		olines=[]
		for index,line in sorted(self.lined.items()):
			#stanzanum=index[-2] if len(index)>1 else 0
			#if stanzanumnow!=stanzanum: olines+=[u'']
			#stanzanumnow=stanzanum
			olines+=[line]
		return u"\n".join(olines)

	@property
	def txt(self):
		if not self.genn: self.gen()
		stanzanumnow=0
		olines=[]
		for stanzatype in ['stanza','versepara']:
			stanzas=list(self.dom.find_all('div',{'type':stanzatype}))
			if len(stanzas): break
		else:
			stanzas=[self.dom]
		
		for stanza in stanzas:
			stanzanumnow+=1
			#if len(stanzas)>1: olines+=['\n'+str(stanzanumnow)]
			for line in stanza.find_all('div',{'type':'line'}):
				olines+=[line.text]
		return u"\n".join(olines)

	def txt_limit(self,n=100):
		txt=self.txt
		return u"\n".join(txt.split(u"\n")[:n])

	@property
	def linelengths(self):
		if not hasattr(self,'_linelengths'):
			self._linelengths=dx={}
			for lineid,line in sorted(self.prosodic.items()):
				dx[lineid]=line.numSyllables
		return self._linelengths

	@property
	def linelengths_bybeat(self):
		if not hasattr(self,'_linelengths_bybeat'):
			self._linelengths_bybeat=dx={}
			for lineid,line in sorted(self.prosodic.items()):
				if not line.numBeats: continue
				print lineid,line.parse_str(), line.numBeats
				dx[lineid]=line.numBeats
		return self._linelengths_bybeat		
	
	@property
	def linelength(self):	# median line length
		if not hasattr(self,'_linelength'):
			self._linelength=pystats.median(self.linelengths.values())
		return self._linelength
	
	@property
	def rhymed(self):
		self.rhyme_net()
		odx={'rhyme_scheme2':self.rhyme_scheme, 'rhyme_scheme_accuracy2':self.rhyme_scheme_accuracy, 'rhyme_weight_avg':self.rhyme_weight_avg}
		#print self.rime_ids
		for k,v in self.discover_rhyme_scheme(self.rime_ids).items(): odx[k]=v
		#for k,v in self.rhyme_clusters.items():
			#odx['rhyme_cluster_'+''.join([str(kx) for kx in k])]=v
		#odx['rhyme_clusters']=self.rhyme_clusters
		return odx

	@property
	def rhymed0(self):
		rimes=[]
		rime_objs=[]

		#for lineid,line in sorted(self.lined.items()):
		#	print [lineid,line]

		for lineid,line in sorted(self.prosodic.items()):
			syllables=line.syllables()
			if not syllables: continue
			print lineid,'\t',self.lined[lineid]
			#last_syllable = [syll for syll in syllables if syll.phonemes()][-1]
			last_syllable=syllables[-1]
			last_rimes=[rime for rime in last_syllable.rimes() if rime.phonemes()]
			if not last_rimes: last_rimes=[syllables[-1].children[0]]
			#print syllables
			#print last_syllable
			#print last_syllable.children[0].children
			assert len(last_rimes) == 1
			last_rime=last_rimes[0]
			rime_objs+=[last_rime]
			last_rime_phonemes_str=last_rime.phonstr()
			rimes+=[last_rime_phonemes_str]

		W=4
		rhymeds=[]
		rime2id={}
		rime_ids=[]
		for i,rime in enumerate(rimes):
			prev_rimes=rimes[i-W if i-W > 0 else 0:i]
			next_rimes=rimes[i+1:i+1+W]
			
			prev_levenshteins = [pytxt.levenshtein(rime,x) for x in prev_rimes]
			next_levenshteins = [pytxt.levenshtein(rime,x) for x in next_rimes]

			it_rhymes=rime in prev_rimes or rime in next_rimes
			#it_rhymes=True in [x<=1 for x in prev_levenshteins] or True in [x<=1 for x in next_levenshteins]

			print i+1,prev_rimes[-1] if prev_rimes else prev_rimes,rime,next_rimes[0] if next_rimes else next_rimes,it_rhymes

			if it_rhymes:
				rhymed=1
				if rime in rime2id:
					rimeid_now=rime2id[rime]
				elif not rime2id:
					rimeid_now=1
					rime2id[rime]=rimeid_now
				else:
					max_in_dict_now=max( rime2id.values() )
					rimeid_now=max_in_dict_now+1
					rime2id[rime]=rimeid_now
				rime_ids+=[rimeid_now]
			else:
				rhymed=0
				rime_ids+=[0]

			rhymeds += [ rhymed ]

			#print prev_rimes,rime,next_rimes,rhymed
			#print

		self.rimes=rimes
		self.rime_objs=rime_objs
		self.rime_ids=rime_ids

		od={}
		od['rhymeness']=pystats.mean(rhymeds)
		od['rhyme_scheme_nums']=rime_ids
		if od['rhymeness']>0.2:
			for k,v in self.discover_rhyme_scheme(rime_ids).items():
				od[k]=v
		else:
			for k,v in self.discover_rhyme_scheme(None).items():
				od[k]=v

		print ">> RHYME SCHEME:",od['rhyme_scheme']

		return od

	@property
	def stanzas(self):
		s2i={}
		for li in sorted(self.lined.keys()):
			s=tuple(li[1:])
			if not s in s2i: s2i[s]=[]
			s2i[s]+=[li]
		return [l for s,l in sorted(s2i.items())]

	@property
	def stanzas_prosodic(self):
		s2i={}
		for li in sorted(self.prosodic.keys()):
			if not self.prosodic[li].words(): continue
			s=tuple(li[1:])
			if not s in s2i: s2i[s]=[]
			s2i[s]+=[li]
		return [l for s,l in sorted(s2i.items())]


	def rhyme_net(self):
		W=4
		import networkx as nx
		G=nx.DiGraph()
		tried=set()
		old=[]
		for stnum,stanza in enumerate(self.stanzas_prosodic):
			for i,lineid1 in enumerate(stanza):
				prev_lines=stanza[i-W if i-W > 0 else 0:i]
				next_lines=stanza[i+1:i+1+W]
				for lineid2 in prev_lines + next_lines:
					#print stnum,lineid1,lineid2
					line1=self.prosodic[lineid1]
					line2=self.prosodic[lineid2]
					node1=str(lineid1[0]).zfill(6)+': '+self.lined[lineid1]
					node2=str(lineid2[0]).zfill(6)+': '+self.lined[lineid2]
					dist=line1.rime_distance(line2)
					
					odx={'node1':node1,'node2':node2, 'dist':dist, 'lineid1':lineid1, 'lineid2':lineid2}
					old+=[odx]
					G.add_edge(node1,node2,weight=dist)

		## ASSIGN RIME IDS
		self.rime_ids=ris=[]
		node2num={}
		nnum=1
		overlaps=set()
		weights=[]
		for node in sorted(G.nodes()):
			#print "NODE",node
			neighbors=sorted(G.edge[node].keys(),key=lambda n2: G[node][n2]['weight'])
			#neighbors=[n for n in neighbors if n>node]
			closest_neighbor=neighbors[0]
			closest_weight=G[node][closest_neighbor]['weight']
			weights+=[closest_weight]
			if closest_weight > 4:
				#nodenum=nnum
				nodenum=0
				node2num[node]=nodenum
				nnum+=1
			if node in node2num:
				nodenum=node2num[node]
			elif closest_neighbor in node2num:
				nodenum=node2num[closest_neighbor]
				node2num[node]=nodenum
				#if n in node2num:
				#	nodenum=node2num[n]
				#	break
			else:
				node2num[node]=nnum
				node2num[closest_neighbor]=nnum
				nodenum=nnum
				nnum+=1

			print node,'\t',nodenum
			G.node[node]['rime_id']=nodenum
			ris+=[nodenum]
			"""print 'closest neighbor:',closest_neighbor
			for x in sorted(G.edge[node].keys(),key=lambda n2: G[node][n2]['weight']):
				print '\t',x,G[node][x]['weight']
			print"""

		"""for i1,ri1 in enumerate(ris):
			for i2,ri2 in enumerate(ris):
				if i2<=i1: continue
				if ri1==ri2:# and ri1!=0:
					for bigram in pytxt.bigrams(range(i1,i2+1)):
						overlaps|={bigram}
		
		## FIND NATURAL BREAKS
		clusters=[]
		cluster=[]
		for i1,i2 in pytxt.bigrams(range(len(ris))):
			if not cluster: cluster+=[i1]
			if (i1,i2) in overlaps:
				cluster+=[i2]
			else:
				clusters+=[cluster]
				cluster=[]
		if cluster: clusters+=[cluster]
		#if not clusters: clusters=[[ci] for ci in range(len(ris))]
		clusters=[tuple(transpose([ris[ci] for ci in cluster])) for cluster in clusters]
		#only_unique = len(set(clusters))==len(clusters)
		#clusters=[c for c in clusters if clusters.count(c)>1 or only_unique]

		## TEMP
		#clusters = [tuple(transpose(c)) for c in pytxt.slice(ris,slice_length=4,runts=False)]

		cluster_freqd={}
		for ctype in clusters:
			#c_ids=[ris[ci] for ci in cluster]
			#ctype=tuple(transpose(c_ids))
			if not ctype in cluster_freqd: cluster_freqd[ctype]=0
			cluster_freqd[ctype]+=len(cluster)
		sumdist=float(sum(cluster_freqd.values()))
		for k,v in cluster_freqd.items(): cluster_freqd[k]=v/sumdist

		self.rhyme_scheme=sorted(cluster_freqd,key=lambda k: -cluster_freqd[k])[0]
		self.rhyme_scheme_accuracy=cluster_freqd[self.rhyme_scheme]
		print ">> RHYME SCHEME 2:",self.rhyme_scheme
		print cluster_freqd
		self.rhyme_clusters=cluster_freqd
		self.rhyme_weight_avg=pystats.mean(weights)"""
		self.rhyme_scheme=''
		self.rhyme_scheme_accuracy=''
		self.rhyme_weight_avg=pystats.mean(weights)





		self.rhymeG=G

		#self.discover_rhyme_scheme(self.rime_ids)

		
		"""weights=[d['weight'] for a,b,d in sorted(G.edges(data=True))]
		weights_z=pystats.zfy(weights)
		for i,(a,b,d) in enumerate(sorted(G.edges(data=True))):
			G.edge[a][b]['weight']=weights_z[i]

		nx.write_gexf(G, 'rhyme_net.gexf')
		old.sort(key=lambda _d: (_d['lineid1'], _d['dist']))
		pytxt.write2('rhyme_net.xls', old)"""



	def discover_rhyme_scheme(self,rime_ids):
		odx={'rhyme_scheme':None, 'rhyme_scheme_accuracy':None, 'rhyme_schemes':None}
		if not rime_ids: return odx

		def translate_slice(slice):
			#slice=[x if slice.count(x)>1 else 0 for x in slice]
			unique_numbers=set(slice)
			unique_numbers_ordered=sorted(list(unique_numbers))
			for i,number in enumerate(slice):
				if number==0: continue
				slice[i] = unique_numbers_ordered.index(number) + 1
			return slice

		def scheme2edges(scheme):
			import pystats
			id2pos={}
			for i,x in enumerate(scheme):
				# x is a rhyme id, i is the position in the scheme
				if x==0: continue
				if not x in id2pos: id2pos[x]=[]
				id2pos[x]+=[i]

			rhymes=[]
			for x in id2pos:
				if len(id2pos[x])>1:
					for a,b in pystats.product(id2pos[x], id2pos[x]):
						if a>=b: continue
						rhymes+=[(a,b)]

			return rhymes

		def test_edges(scheme_exp,scheme_obs):
			import pystats
			edges_exp=scheme2edges(scheme_exp)
			edges_obs=scheme2edges(scheme_obs)
			#print scheme_exp,scheme_obs

			#return pystats.mean([int(e in edges_obs) for e in set(edges_exp)])
			import distance
			dist=distance.nlevenshtein(scheme_exp, scheme_obs)
			#for s1,s2 in zip(scheme_exp,scheme_obs):
			#	if 0 in [s1,s2] and {s1,s2}!={0}:
			#		dist+=2
			print scheme_exp,'\t',scheme_obs,'\t',dist
			#if scheme_exp==(1,1) and scheme_obs!=(1,1):
			#	dist+=2
			dist = dist * 10**(1/len(scheme_exp))
			return dist

		"""def test_scheme2(scheme):
			print "TESTING SCHEME:",scheme
			scheme_nums=scheme2nums(scheme)
			rime_is=[i for i in range(len(self.rimes))]
			line_ids=sorted(self.lined.keys())
			stanzanum2rime_is={}
			for ri in rime_is:
				stnum=line_ids[ri][1]
				if not stnum in stanzanum2rime_is: stanzanum2rime_is[stnum]=[]
				stanzanum2rime_is[stnum]+=[ri]

			for stanzanum,rime_is in sorted(stanzanum2rime_is.items()):
				slices=pytxt.slice(rime_is,slice_length=len(scheme_nums),runts=False)
				lines=[line for lineid,line in sorted(self.lined.items())]
				dists=[]
				for slicei,slicex in enumerate(slices):
					slice_rimes=[self.rime_objs[x] for x in slicex]
					snum2rime={}

					print slicei
					print slicex
					for x in slicex:
						print line_ids[x], lines[x]



					for snum,srime in zip(scheme_nums,slice_rimes):
						if not snum: continue
						if not snum in snum2rime: snum2rime[snum]=[]
						snum2rime[snum]+=[srime]

					print snum2rime

					s_dists=[]
					for snum,srimes in snum2rime.items():
						srimes_is=list(range(len(srimes)))
						for r1,r2 in pystats.product(srimes_is,srimes_is):
							if r1<=r2: continue
							r1,r2=srimes[r1],srimes[r2]
							dist=r1.phonetic_distance(r2,normalized=True)
							print "COMPARING",r1.phonstr(),"AND",r2.phonstr()
							print "DIST = ",dist
							print
							s_dists+=[dist]
					print
					dists+=s_dists

			return sum([sd**2 for sd in dists])"""










		def test_scheme(scheme):
			#print "scheme:",scheme
			scheme_nums=scheme2nums(scheme)
			slices=pytxt.slice(rime_ids,slice_length=len(scheme_nums),runts=True)
			matches=[]
			did_not_divide=0
			#print ">> RIME IDS:",rime_ids
			for si,slice in enumerate(slices):
				tslice=translate_slice(slice)
				#match = int(tslice == scheme_nums)
				match = test_edges(scheme_nums[:len(tslice)], tslice)
				if len(scheme_nums) != len(tslice):
					did_not_divide=1
				#print ">>",si,"slice, looking for:",scheme_nums,"and found:",tslice
				#print ">> MATCH:", match
				#print
				matches+=[match]
				# translate down

			#print matches
			match_score=pystats.median(matches) + did_not_divide*10
			return match_score

		#test_scheme('abab',rime_ids)
		if False:
			couplet_accuracy=test_scheme('aa')
			if couplet_accuracy > 0.5:
				odx['rhyme_scheme']='aa'
				odx['rhyme_scheme_accuracy']=couplet_accuracy
			else:
				odx['rhyme_scheme']='Sonnet'
		else:
			scheme_scores={}
			for schemed in RHYME_SCHEMES:
				scheme=schemed['scheme']
				scheme_score=test_scheme(scheme)
				#if scheme_score:
				scheme_scores[scheme]=scheme_score

			odx['rhyme_schemes']=scheme_scores
			#for scheme,scheme_score in sorted(scheme_scores.items(),key=lambda lt: -len(lt[0])):
			for scheme,scheme_score in sorted(scheme_scores.items(),key=lambda lt: -lt[1]):
				print scheme, scheme_score
			
			for scheme,scheme_score in sorted(scheme_scores.items(),key=lambda lt: (lt[1],-len(lt[0]))):
				odx['rhyme_scheme']=scheme
				odx['rhyme_scheme_accuracy']=scheme_score
				break

		if not odx['rhyme_scheme']: odx['rhyme_scheme']='Unknown'
		return odx




	@property
	def rhymescheme(self,rhyme_types=[1]):
		letters='abcdefghijklmnopqrstuvwxyz'
		newletter=letters[0]
		rimed={}   # rime 2 lineid
		scheme={}  # lineid 2 letter
		scheme2={} # lineid 2 rime (active)
		for lineid,line in sorted(self.prosodic.items()):
			sylls=line.syllables()
			line_rimes=[]
			for n in rhyme_types:
				last_sylls=sylls[-n:]
				rime=last_sylls[0].rimes()[0].phonemes()
				for syll in last_sylls[1:]:
					rime+=syll.phonemes()
				rime=tuple(rime)
				line_rimes+=[rime]
				print lineid, rime
			
			for rime in line_rimes:
				if rime in rimed:
					letter = scheme[rimed[rime]]
					rime_active = rime
					break
			else:
				letter = newletter
				rime_active = line_rimes[0]
				newletter = letters[letters.index(newletter)+1]
			
			scheme[lineid] = letter
			
			for rime in line_rimes: rimed[rime]=lineid
			
			print lineid, letter, line.lines()
			print
	
	
	@property
	def rhymescheme_basedonstress(self):
		letters='abcdefghijklmnopqrstuvwxyz'
		newletter=letters[0]
		rimed={}   # rime 2 lineid
		scheme={}  # lineid 2 letter
		scheme2={} # lineid 2 rime (active)
		for lineid,line in sorted(self.prosodic.items()):
			sylls=line.syllables()
			rime_sylls=[]
			for syll in reversed(sylls):
				rime_sylls.insert(0,syll)
				if syll.feature('+prom.stress'): break
			print line.lines(), rime_sylls
		
		
			
	
	def txt_old(self,clean=True):
		if not clean:
			return "\n".join(self.lines)
		return "\n".join([l.split('\n')[0] for l in self.lines]).replace('&indent;','       ')

	
	def search(self,wordlist,posonly=True,sumpre='sum_'):
		if not self.genn: self.gen()
		results=[]
		
		#sumnow=0
		for lnum in range(len(self.lines)):
			line=self.lines[lnum].strip().lower()
			lwords=line.split()
			lset=set(lwords)
			
			result={'countsum':0}
			
			for word in wordlist:
				wordset=set(word.lower().strip().split())
				
				if len(lset&wordset)==len(wordset):
					c=1
				else:
					c=0
				
				result['count_'+word]=c
				if posonly and c==0:
					continue				

				result['countsum']+=c
				
			
			if posonly and result['countsum']==0:
				continue
				
			result['type']='Line'
			result['label']=self.lines[lnum].strip()
			result['id']=str(self.id)+".ln"+str(lnum)
			result['linenum']=lnum
			result['poemid']=str(self.id)
			result['poemtitle']=str(self.title)
			results.append(result)
		
		if sumpre:
			sums={}
			for x in results:
				for k,v in x.items():
					if not k.startswith('count_') and k!='countsum': continue
					try:
						x[sumpre+k]=sums[k]+v
						sums[k]+=v
					except KeyError:
						sums[k]=v
						x[sumpre+k]=v
		
		return results
	
	@property
	def isSonnet(self):
		if len(self.lined)!=14: return False
		if not int(self.linelength) in [9,10,11]: return False
		return True
	
	@property
	def hasVariableLineLengths(self,threshold=0.70):
		linelengths=[(ll if not ll%2 else ll-1) for ll in self.linelengths.values()]
		linedistro=pytxt.toks2freq(linelengths,tfy=True)
		max_ll=max(linedistro.values())
		return max_ll<threshold
		
	
	def extract_lines(self,remove_epigrams=False):
		def get_distance(tag,dist=[]):
			if tag.parent.name=='body': return dist
			if tag.parent.name=='div' and tag.parent.get('type','')!='firstline':
				prev_siblings=[pv for pv in list(tag.parent.previous_siblings) if hasattr(pv,'name')]
				dist.insert(0,len(prev_siblings)+1)
			return get_distance(tag.parent,dist)
			
		lined={}
		for i,line in enumerate(self.dom.find_all("div",{'type':'line'})):
			if line.find_parent('div',{'type':'epigraph'}): continue
			linenum=line.get('n',None)
			linenum=int(linenum) if linenum else i+1
			#lineid=tuple([i+1] + get_distance(line,dist=[]) + [linenum])

			stanza=line.find_parent('div',{'type':'stanza'})
			stanzanum=0 if not stanza else stanza['n']
			lineid=(i+1, stanzanum)
			linetext=line.text.replace('\n',' ').replace('  ',' ')
			## greek char fix
			if '&gr' in linetext:
				for ent in pytxt.yanks(linetext,('&gr',';')):
					linetext=linetext.replace('&gr'+ent+';',ent)
			
			linetext=linetext.replace('& ','and ')
			linetext=linetext.replace(u'—',' ')
			linetext=linetext.replace(u'&ebar;','e')
			lined[lineid]=linetext
		
		## clean lined ###
		
		# remove epigrams, etc
		if remove_epigrams:
			lenfreq=pytxt.toks2freq([len(lineid) for lineid in lined.keys()])
			mostcommonlen=[_k for _k in sorted(lenfreq,key=lambda __k: -lenfreq[__k])][0]
			if len(lenfreq)>1:
				## remove other lengths
				print ">> removing lines of non-standard positions within poem [eg epigraphs]. Line depth frequencies:",lenfreq
				for _k in lined.keys():
					if len(_k)!=mostcommonlen:
						del lined[_k]
		
		"""
		### re-number lines
		## HACK
		newlined={}
		for i,lid in enumerate(sorted(lined.keys())):
			newlined[i+1]=lined[lid]
			del lined[lid]
		return newlined
		###
		"""
		return lined
		"""
		# return if there is only one number in the index
		if mostcommonlen<2: return lined

		## convert to dictionary
		numd={}
		for lid in sorted(lined.keys()):
			dnow=numd
			for x in lid[1:]:
				if not x in dnow: dnow[x]={}
				dnow=dnow[x]
		
		def renumber(numd,nums=[],path=[]):
			if not numd:
				nums+=[path]
				return nums
			
			for i,key in enumerate(sorted(numd.keys())):
				numd['_'+str(i+1)]=numd[key]
			
			for key in numd.keys():
				if type(key)!=str:
					del numd[key]
			
			for key in numd.keys():
				realkey=int(key[1:])
				numd[realkey]=numd[key]
				del numd[key]
			
			for i,key in enumerate(sorted(numd)):
				nums=renumber(numd[key],nums=nums,path=path+[key])
			
			return nums

		lines_renumbered=renumber(numd)
		newlined={}
		for i,lid in enumerate(sorted(lined.keys())):
			print i, lid, len(lines_renumbered)
			newlid=tuple([i+1]+lines_renumbered[i])
			newlined[newlid]=lined[lid]
			del lined[lid]
		return newlined
		"""

def standardize(line):
	import re, urlparse, pytxt
	words=line.split()
	for i,word in enumerate(line.split()):
		p0,word,p1=pytxt.gleanPunc2(word)
		p0,word,p1=unicode(p0),unicode(word),unicode(p1)
		if word and (not "'" in word):
			initial_cap=word[0].upper()==word[0]
			all_cap=word.upper()==word
			word=unicode(spelling_d.get(word.lower(),word)) # send lower case version to spelling standardizer
			if all_cap:
				word=word.upper()
			elif initial_cap:
				word=word[0].upper()+word[1:]
		
		words[i]=unicode(p0)+unicode(word)+unicode(p1)
	line=' '.join(words)
	return line


def openmary(line,lang='en_US',fix_contractions=True):
	import re, urlparse, pytxt

	def urlEncodeNonAscii(b):
		return re.sub('[\x80-\xFF]', lambda c: '%%%02x' % ord(c.group(0)), b)

	def iriToUri(iri):
		parts= urlparse.urlparse(iri)
		return urlparse.urlunparse(
			part.encode('idna') if parti==1 else urlEncodeNonAscii(part.encode('utf-8'))
			for parti, part in enumerate(parts)
		)

	import urllib2,pytxt
	#line=pytxt.ascii(line).replace(' ','+')
	line=line.replace(' ','+')
	link=u'http://localhost:59125/process?INPUT_TEXT={0}&INPUT_TYPE=TEXT&OUTPUT_TYPE=ALLOPHONES&LOCALE={1}'.format(line,lang)
	try:
		f=urllib2.urlopen(iriToUri(link))
	except urllib2.HTTPError:
		print "!? failed on:",line
		print link
		print iriToUri(link)
		return ''

	t=f.read()
	f.close()
	
	### OPEN MARY CLEANING OPERATIONS
	import bs4
	xml=bs4.BeautifulSoup(t)
	
	## fix word string problem
	for word in xml.find_all('t'): word['token']=word.text.strip()

	## CONTRACTION FIX
	if fix_contractions:
		for para in xml.find_all('p'):
			for phrase in para.find_all('phrase'):
				for word1,word2 in pytxt.bigrams(phrase.find_all('t')):
					w2text=word2.text.strip().lower()
					if w2text.startswith("'") and word2.find_all('syllable') and word1.find_all('syllable'):
						phones2add=word2.find_all('syllable')[-1]['ph'].strip()
						word1.find_all('syllable')[-1]['ph']+=' '+phones2add
						word1['token']+=w2text
						word2.decompose()
	
	t=xml.prettify()
	#####
	
	return t




class PoemTXT(Poem):
	def __init__(self,txt,id=None,title=None):
		self.id=pytxt.now() if not id else id
		print ">> gen {0} from TXT...".format(self.id)

		self.dbfn=None
		txt=txt.replace('\r\n','\n').replace('\r','\n')
		while '\n\n\n' in txt: txt=txt.replace('\n\n\n','\n\n')
		self.title=title if title else [l.strip() for l in txt.split('\n') if l.strip()][0]


		lined={}
		linenum=0
		stanza_lens=[]
		for stanza_i,stanza in enumerate(txt.split('\n\n')):
			stanza=stanza.strip()
			num_line_in_stanza=0

			for line_i,line in enumerate(stanza.split('\n')):
				line=line.strip()
				if not line: continue
				num_line_in_stanza+=1
				linenum+=1
				stanzanum=stanza_i+1
				lineid=(linenum, stanzanum)
				linetext=line
				linetext=linetext.replace('& ','and ')
				linetext=linetext.replace(u'—',' ')
				linetext=linetext.replace(u'&ebar;','e')
				lined[lineid]=linetext

			stanza_lens+=[num_line_in_stanza]

		self._stanza_length=stanza_lens[0] if len(set(stanza_lens))==1 else None

		
		self.lined=lined
		self.numLines=len(self.lined)
		#self.lines=[line for index,line in sorted(self.lined.items())]	
		print "\t>> done."
		self.genn=True




def transpose(slice):
	unique_numbers=set(slice)
	unique_numbers_ordered=sorted(list(unique_numbers))
	for i,number in enumerate(slice):
		if number==0: continue
		slice[i] = unique_numbers_ordered.index(number) + 1
	return slice

def scheme2nums(scheme):
	scheme=scheme.replace(' ','')
	scheme_length=len(scheme)
	alphabet='abcdefghijklmnopqrstuvwxyz'
	scheme_nums=[alphabet.index(letter)+1 if scheme.count(letter)>1 else 0 for letter in scheme]
	return scheme_nums

def transpose_up(slice):
	import string
	return ''.join(string.ascii_lowercase[sx-1] if sx else 'x' for sx in slice)


def schemenums2dict(scheme):
	d={}
	for i,x in enumerate(scheme):
		for ii,xx in enumerate(scheme[:i]):
			if x==xx:
				d[i]=ii
	return d