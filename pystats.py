from __future__ import division
"""
Calculate mean and standard deviation of data x[]:
    mean = {\sum_i x_i \over n}
    std = sqrt(\sum_i (x_i - mean)^2 \over n-1)
"""
from decimal import *
getcontext().prec = 20
from math import *

def ndian(numericValues,n=2):
	theValues = sorted(numericValues)
	if len(theValues) % n == 1:
		return theValues[int((len(theValues)+1)/n)-1]
	else:
		lower = theValues[int(len(theValues)/n)-1]
		upper = theValues[int(len(theValues)/n)]
		return (float(lower + upper)) / n

def gtest(obs,exp):
	pass

def herfindahl(l,tfy=True):
	if not sum(l): return 0
	if tfy:
		summ=sum(l)
		l=[x/summ for x in l]
	return sum([x*x for x in l])


def median(numericValues):
	try:
		import numpy as np
		return np.median(numericValues)
	except:
		return ndian(numericValues,n=2)

def iqr(numericValues):
	return upperq(numericValues) - lowerq(numericValues)

def mean_stdev(x):
    if not len(x): return 0,0
    if len(x)==1: return x[0],0
    from math import sqrt
    n, mean, std = len(x), 0, 0
    for a in x:
	mean = mean + a
    mean = mean / float(n)
    for a in x:
	std = std + (a - mean)**2
    std = sqrt(std / float(n-1))
    return mean, std

def mean(x):
	return mean_stdev(x)[0]

def correlate(x,y):
	from statlib.stats import pearsonr
	return pearsonr(x,y)

def correlate_dictionaries(x,y):
	xkeys=set(x.keys())
	ykeys=set(y.keys())
	keys=xkeys&ykeys		# intersection
	
	xx=[v for k,v in sorted(x.items()) if (k in keys)]
	yy=[v for k,v in sorted(y.items()) if (k in keys)]
	
	return correlate(xx,yy)

def str_float(float):
	return str(Decimal(str(float)))

def zfy(tfdict,limit=None):
	if type(tfdict)==type({}):
		mean,stdev=mean_stdev(tfdict.values())
		
		zdictz={}
		for k,v in tfdict.items():
			if not stdev:
				zdictz[k]=0
			else:
				zdictz[k]=(v-mean)/stdev
			if limit:
				if zdictz[k]>limit:
					del zdictz[k]
			
		return zdictz
	elif type(tfdict)==type([]):
		mean,stdev=mean_stdev(tfdict)
		zdictz=[]
		for v in tfdict:
			if not stdev:
				score=0
			else:
				score=(v-mean)/stdev
			if limit and score>limit:
				continue
			zdictz+=[score]
		return zdictz
	else:
		return tfdict

def tfy(tfdict,limit=None):
	if type(tfdict)==type({}):
		#mean,stdev=mean_stdev(tfdict.values())
		summ=sum(tfdict.values())
		
		zdictz={}
		for k,v in tfdict.items():
			zdictz[k]=v/summ
			if limit:
				if zdictz[k]>limit:
					del zdictz[k]

		return zdictz
	elif type(tfdict)==type([]):
		#mean,stdev=mean_stdev(tfdict)
		summ=sum(tfdict)
		
		zdictz=[]
		for v in tfdict:
			score=v/summ
			if limit and score>limit:
				continue
			zdictz+=[score]
		return zdictz
	else:
		return tfdict


have_numpy = False
try:
	import numpy as np
	have_numpy = True
except ImportError:
	pass


def gini(x, use_numpy=True): #follow transformed formula
	"""Return computed Gini coefficient.
	:contact: aisaac AT american.edu
	"""
	xsort = sorted(x) # increasing order
	if have_numpy and use_numpy:
		y = np.cumsum(xsort)
	else:
		y = cumsum(xsort)
	B = sum(y) / (y[-1] * len(x))
	return 1 + 1./len(x) - 2*B


def entropy(l,perc_ideal=False):
	import math
	if type(l) in [str,unicode]: l=[c for c in l]
	if type(l) in [dict]:
		length=sum(l.values())		
		prob=[float(px)/length for px in l.values()]
	else:
		# get probability of ents in list
		length=len(l)		
		prob = [ float(l.count(c)) / length for c in set(l) ]
		
	# calculate the entropy
	entropy = - sum([ p * math.log(p) / math.log(2.0) for p in prob ])

	if perc_ideal:
		prob = 1.0 / length
		ideal_entropy=-1.0 * length * prob * math.log(prob) / math.log(2.0)
		return entropy/ideal_entropy

	return entropy



def shannon(l):
	return shannon_evenness(l)

def shannon_evenness(l):
	"""Returns the SHANNON-WIENER DIVERSITY INDEX for a distribution."""

	S=len(l)	# number of species / species richness
	N=sum(l)	# number of individuals
	h0=0

	l.sort(reverse=True)
	if len(l)<2: return 0
	
	for i in l:
		if i<=0: break
		pi = i/N	# relative abundance of species (i)
		h0+=(pi*log(pi))

	h0 = h0*(-1)
	r= h0/log(S)
	return r

def shannon_diversity(l):
	"""Returns the SHANNON-WIENER DIVERSITY INDEX for a distribution."""

	S=len(l)	# number of species / species richness
	N=sum(l)	# number of individuals
	h0=0

	l.sort(reverse=True)
	if len(l)<2: return 0
	
	for i in l:
		if i<=0: break
		pi = i/N	# relative abundance of species (i)
		h0+=(pi*log(pi))

	h0 = h0*(-1)
	r=h0
	return r



# 
# def spine(ld,xname='year',yname='tf'):
# 	spine={}
# 	for x in ld:
# 		spine[x[xname]]=x[yname]
# 	return spine


	
def plotbooktfs(tfs,fn,title=''):
	qdict={'Year':[],'AggregateTermFrequency':[],'corpus':[]}
	
	if 'z.' in fn:
		isz=True
	else:
		isz=True
		
	if not isz:
		zdict=zfy(tfs)
		#plotbooktfs(zdict,fn.replace('.png','.z.png'))
	
	for novel,score in tfs.items():
		if not isz:
			if abs(zdict[novel])>10: continue
		#else:
		#	if abs(score)>10: continue
		
		try:
			#if not score: continue
			#print novel.corpus
			qdict['Year'].append(novel.year)
			qdict['AggregateTermFrequency'].append(score)
			
			#if novel.genre=="Gothic":
			#	qdict['corpus'].append(1)
			#else:
			#	qdict['corpus'].append(2)
			qdict['corpus'].append(1)
		except AttributeError:
			if type(novel)==type(""):
				qdict['Year'].append(int(novel.split(".")[0]))
			else:
				qdict['Year'].append(novel)
			
			qdict['AggregateTermFrequency'].append(score)
			
			if type(novel)==type("") and ("shakespeare" in novel.lower()):
				qdict['corpus'].append(1)
			else:
				qdict['corpus'].append(0)
		
	if len(qdict['Year'])!=len(qdict['AggregateTermFrequency']):
		return None
	try:
		scatterplot(qdict,fn,x='Year',y='AggregateTermFrequency',title=title)
	except:
		return None

def dist(xvals,yvals):
	import math
	return math.sqrt( (xvals[1]-xvals[0])**2 + (yvals[1]-yvals[0])**2 )

def getdimensions(datadict):
	ncol=len(datadict.keys())
	nrow=len(datadict[datadict.keys()[0]])
	return (nrow,ncol)

def getCols(ld,allcols=False):
	cols=[]
	for row in ld:
		if not type(row)==type({}): continue
		if not len(row.keys()): continue
		
		if not len(cols):
			cols=row.keys()
			continue
		
		if allcols:
			for x in set(row.keys())-set(cols):
				#print ">> adding col: "+x
				cols.append(x)
		else:
			for x in set(cols)-set(row.keys()):
				#print cols
				#print row.keys()
				
				#print ">> removing col: "+x
				cols.remove(x)
		
	return cols
	

def dataframe(ld,cols=None,rownamecol=None,factorcol=None,thresholds=[],allcols=False,trimbyVariance=True,factor=True,rank=False,sumcol='',z=False,df=True,zero=0.0):
	import rpy2.robjects as ro
	dd={}
	if not cols:
		cols=getCols(ld,allcols)
	
	
	if type(ld)!=type([]):
		print cols
		print list(ld.colnames)
		
		## type is dataframe
		dd={}
		for col in cols:
			print col
			if col==factorcol: continue
			print ld.colnames
			print ld.colnames.index(col)
			dd[col]=ld[list(ld.colnames).index(col)][0]
		
			
		if factorcol:
			return (ro.DataFrame(dd), ld[list(ld.colnames).index(factorcol)][0])
		else:
			return ro.DataFrame(dd)
			
	
	
	if rownamecol:
		if rownamecol in cols:
			cols.remove(rownamecol)
	
	#print cols
		

	
	
	
	if type(allcols)==type(1):
		keysums={}
		for col in cols:
			keysums[col]=[]
		for x in ld:
			for k in x:
				value=x[k]
				if type(value)==type(float()) or type(value)==type(1):
					try:
						keysums[k]+=[value]
					except KeyError:
						continue
		from operator import itemgetter
		i=0
		allowedkeys=[]
		
		for k in keysums:
			if len(keysums[k])<=1:
				keysums[k]=0
				continue
			if trimbyVariance:
				keysums[k]=sum([abs(x) for x in zfy(keysums[k])])
			else:
				keysums[k]=sum(keysums[k])
		
		
		sumvariances_df=sum(keysums.values())
		sumvariances=0.0	
		
		for key,score in sorted(keysums.items(),key=itemgetter(1),reverse=True):
			print key,"\t",score
			i+=1
			if i>allcols:
				break
			sumvariances+=score
			allowedkeys.append(key)
		#print allowedkeys
		cols=[]
		for i in range(len(allowedkeys)):
			if rank:
				k='r'+str(i+1).zfill(len(str(len(allowedkeys))))+'.'+allowedkeys[i]
			cols.append(k)

		#cols=allowedkeys
		#print cols
		print ">> sum of Z-variances in dataset:", sumvariances_df
		print ">> sum of Z-variances loaded:", sumvariances
		print ">> (ratio) sum of loaded Z-variances / sum of possible:", sumvariances/sumvariances_df
		print ">> # features loaded:", len(cols)
		print ">> (ratio) sum of loaded Z-variances / # of features loaded:", sumvariances/len(cols)
		print
		#print cols
	
	
	for i in range(len(cols)):
		k=cols[i]
		dd[k]=[]
	if sumcol:
		dd['summ']=[]

	rownames=[]
	
	
	
	for x in ld:
		if len(thresholds):
			tocontinue=False
			for (colname,limit) in thresholds:
				if float(x[colname])>float(limit):
					tocontinue=True
			if tocontinue:
				continue
		
		for k in cols:
			k_look=k
			k_store=k
			if rank:
				k_look=".".join(k.split(".")[1:])
			try:
				value=x[k_look]
			except KeyError:
				value=zero

			dd[k_store].append(value)
			
		if sumcol:
			if not len(dd['summ']):
				dd['summ'].append(float(x[sumcol]))
			else:
				dd['summ'].append(dd['summ'][-1]+float(x[sumcol]))
	
		if rownamecol:
			rownames.append(x[rownamecol])
	
	
	for k,v in dd.items():
		if type(v[0])==type(''):
			dd[k]=ro.StrVector(v)
			if factor:
				dd[k]=ro.FactorVector(dd[k])

		else:
			if z:
				v=zfy(v)
						
			dd[k]=ro.FloatVector(v)
	
	if df:
		df=ro.DataFrame(dd)
		if rownamecol:
			rownames=ro.FactorVector(rownames)
			#print df.rownames
			df.rownames=rownames
		return df
	else:
		return dd
	

def fits(X,Y,degs=range(1,4)):
	fitld=[]
	mean=mean_stdev(Y)[0]
	ss=sum( (abs(x-mean)**2) for x in Y)
	degd={
		1: 'Linear regression',
		2: 'Binomial regression',
		3: 'Trinomial regression',
		4: 'Quadrinomial regression'
	}
	for deg in degs:
		fitd={}
		fitd['deg']=deg
		if deg in degd: fitd['type']=degd[deg]
		cs, residuals, rank, singular_values, rcond = polyfit(X,Y,deg)
		try:
			fitd['rr']=1-(float(residuals)/ss)
		except TypeError:
			continue
		cs=list(reversed(list(cs)))
		fitd['formula']='Y = '+ ' + '.join( reversed([str(round(cx,2))+'X^'+str(ci) for ci,cx in enumerate(cs)]) )
		fitd['formula']=fitd['formula'].replace('X^1','X').replace('X^0','')
		fitd['data']=[]
		for yr in sorted(list(set(X))):
			fitd['data']+=[ (yr, sum([ cx * pow(yr,ci) for ci,cx in enumerate(cs)])) ]
		fitld+=[fitd]
	return fitld





def fit(ld,ykey):
	import rpy2.robjects as ro
	r = ro.r
	
	keys=set(getCols(ld))
	ykeys=set([ykey])
	xkeys=keys.difference(ykeys)
	
	lm=r['lsfit'](dataframe(ld,xkeys),dataframe(ld,ykeys))
	
	#df=dataframe(ld)
	#frmla=ykey+" ~ "+"+".join(xkeys)
	#lm=r['glm'](formula=frmla,data=df,family='quasibinomial']
	#return "\n\n".join(str(x) for x in [ r['anova'](lm,test='Chisq') ] )
	r['ls.print'](lm)
	
def gaussian():
	pass





	
def polyfit(xs,ys,deg=2,norm=True):
	def fwhm(coeffs,ymax):
		pc=coeffs.copy()
		pc[-1]-=ymax
		x_at_max=min(np.roots(pc))
		r1=np.roots(pc).tolist()
		
		pc=coeffs.copy()
		pc[-1]-=ymax/2
		x_at_half_max=min(np.roots(pc))
		r2=np.roots(pc).tolist()

		rx=[abs(rx1-rx2) for rx1,rx2 in product(r1,r2) if rx1!=rx2]
		diff=min(rx)
		return diff * 2




	import numpy as np
	#print ">> fitting word: "+word
	
	X=np.array(xs)
	Y=np.array(ys)
	#if norm:
	#	Ymax=max(Y)
	#	Y=[y/Ymax for y in Y]
	XY=zip(X,Y)
	
	try:
		coeffs=np.polyfit(X,Y,deg)
	except np.linalg.linalg.LinAlgError:
		return 
	polymake=np.poly1d
	p=polymake(coeffs)
	yhat = p(X)                         # or [p(z) for z in x]
	ybar = np.sum(Y)/len(Y)          # or sum(y)/len(y)
	ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
	sstot = np.sum((Y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])

	results={'deg':deg, 'coeffs':coeffs.tolist()}
	results['str']=' + '.join('{0}*x{1}'.format(x,"^"+str(deg-i) if deg-i>1 else '') for i,x in enumerate(coeffs))
	results['a']=results['coeffs'][0]
	results['a_is_pos']=int(results['a']>0)
	results['n_is_even']=int(not deg%2)
	results['max']=max(yhat)
	results['max_real']=max(Y)
	results['x_at_max']=[x for x,y in zip(X,yhat) if y==results['max']][0]
	results['x_at_max_real']=[x for x,y in zip(X,Y) if y==results['max_real']][0]
	results['x_at_max_real_avg3']=mean([x for x,y in sorted(XY,key=lambda _xy: -_xy[-1])][:3])

	results['min']=min(yhat)
	results['min_real']=min(Y)
	results['x_at_min']=[x for x,y in zip(X,yhat) if y==results['min']][0]
	results['x_at_min_real']=[x for x,y in zip(X,Y) if y==results['min_real']][0]
	results['x_at_min_real_avg3']=mean([x for x,y in sorted(XY,key=lambda _xy: _xy[-1])][:3])

	extremestr='min' if results['a_is_pos'] else 'max'
	for k in results.keys():
		if extremestr in k:
			results[k.replace(extremestr,'extreme')]=results[k]

	results['fwhm']=fwhm(coeffs,results['max'])
	results['halflife']=results['fwhm']/2
	results['fit'] = ssreg / sstot
	print results
	return results


def datadict(ld):
	dd={}
	for row in ld:
		for k,v in row.items():
			v=float(v)
			try:
				dd[k].append(v)
			except KeyError:
				dd[k]=[]
				dd[k].append(v)
	return dd


def plots(ld,x=None,y=None,odir='/Lab/Projects/q/output/rplots/',density=True):
	dd=datadict(ld)
	import os
	
	if x or y:
		for k,v in dd.items():
			if x:
				fn=os.path.join(odir,'_X_'.join([x,k])+'.png')
				rplot(dd[x],v,fn,density=density)
			else:
				fn=os.path.join(odir,'_X_'.join([k,y])+'.png')
				rplot(v,dd[y],fn,density=density)
	else:
		for (a,b) in pair(dd.keys()):
			fn=os.path.join(odir,'_X_'.join([a,b])+'.png')
			rplot(dd[a],dd[b],fn,density=density)
			
def corrgram(fn,df,w=1600,h=1600):
	from rpy2.robjects.packages import importr
	from rpy2.robjects import r
	grdevices = importr('grDevices')
	importr('corrgram')
	grdevices.png(file=fn, width=w, height=h)
	
	
	
	#r['corrgram'](df, order=True, lower_panel='panel.shade',upper_panel='NULL', text_panel='panel.txt', main="Car Milage Data (unsorted)")
	#print df
	r['corrgram'](df,lower_panel='panel.shade',upper_panel='panel.pts')
	
	
	grdevices.dev_off()
	print ">> saved: "+fn



def hclust(fn,df,w=1100,h=900,cor=False,dendro=True,k=None):
	from rpy2 import robjects as ro
	from rpy2.robjects.packages import importr
	from rpy2.robjects import r
	
	
	


	if cor:
		
		c=r['cor'](df)
	
		for row_i in xrange(1, c.nrow+1):
		    for col_i in xrange(1, c.ncol+1):
				key=ro.rlc.TaggedList((row_i,col_i))
			
				x=list(c.rx[key])[0]
			
				c.rx[key] = (1-x)/2
	
		dd = r['as.dist'](c)
	else:
		dd= r['dist'](df)
		
	
	hclust=r['hclust'](dd)
	
	if k and type(k)==type(1):
		groups=list(r['cutree'](hclust,k))
		#help(df)
		#print df.__dict__
		rows=list(df.rownames)
		fidelities={}
		fidelities_bycluster={}
		
		for rowclan in set([row.strip().split("_")[0] for row in rows]):
			fidelities[rowclan]={}
			for ii in range(1,k+1):
				gr="Cluster_"+str(ii).zfill(2)
				fidelities[rowclan][gr]=0
		
		for ii in range(1,k+1):
			gr="Cluster_"+str(ii).zfill(2)
			fidelities_bycluster[gr]={}
			for rowclan in set([row.strip().split("_")[0] for row in rows]):
				fidelities_bycluster[gr][rowclan]=0
		
		for i in range(len(groups)):
			row=str(rows[i])
			group="Cluster_"+str(groups[i]).zfill(2)
			rowclan=row.strip().split("_")[0]
			fidelities[rowclan][group]+=1
			fidelities_bycluster[group][rowclan]+=1
		
		
		ld=[]
		ldd=[]
		for rowclan,groupd in fidelities.items():
			if len(groupd)==1:
				entropy=1
			else:
				entropy=shannon(groupd.values())

			ldd.append( {'RowType':rowclan,'ShannonDiversityIndex':entropy} )
			print rowclan,"\t",sorted(groupd.values(),reverse=True)
			for group,count in groupd.items():
				d={}
				d['RowType']=rowclan
				d['Cluster_Num']=group
				d['count']=count
				ld.append(d)

		
				
		df_clusternum=dataframe(ld)
		df_diversity=dataframe(ldd)
		
		print df_diversity
		
		
		
		ld_clusters=[]
		for cluster,cland in fidelities_bycluster.items():
			entropy=shannon(cland.values())
			if not entropy: continue
			ld_clusters.append( {'Cluster_Num':cluster,'ShannonDiversityIndex':entropy} )
			print cluster,"\t",sorted(cland.values(),reverse=True)
		
		df_clusterdiversity=dataframe(ld_clusters)
		print df_clusterdiversity
		
		
		
		plotframe( fn.replace('.png', '.groupdiversity.png'), df_diversity, x='ShannonDiversityIndex', y='RowType', col='RowType', size=5, smooth=False)
		plotframe( fn.replace('.png', '.clusterdiversity.png'), df_clusterdiversity, x='ShannonDiversityIndex', y='Cluster_Num', col='Cluster_Num', size=5, smooth=False)
		plotframe( fn.replace('.png','.groupcounts.png'), df_clusternum, x='count', y='Cluster_Num', col='RowType', jitter=True, size=5, smooth=False)
		
	
	if dendro:
		grdevices = importr('grDevices')
		grdevices.png(file=fn, width=w, height=h)
		r['plot'](hclust)
		grdevices.dev_off()
		print ">> saved: "+fn	
	
def normlistlen(l,index,n=100):
	num=max(index)
	step=num/n
	now=[]
	o=[]
	for i in range(len(l)):
		li=l[i]
		ii=index[i]

		
		if ii>=step or i==len(l)-1:		
			if len(now)==1:
				mean=now[0]
			else:
				try:
					mean,std=mean_stdev(now)
				except ZeroDivisionError:
					mean=o[len(o)-1]
			o+=[mean]
			step+=(num/n)
			now=[]
		else:
			now.append(li)
	
	if len(o)>n:
		o=o[:n]
	elif len(o)<n:
		for i in range(len(o),n):
			o+=[o[len(o)-1]]

	return o
	
	


def rplt(obj,fn,w=800,h=800,xlabel="",ylabel="",label=""):
	from rpy2.robjects.packages import importr
	from rpy2.robjects import r
	grdevices = importr('grDevices')
	grdevices.png(file=fn, width=w, height=h)
	
	r.plot(obj)
	
	grdevices.dev_off()
	print ">> saved: "+fn
	

def rplot(x,y,fn,w=800,h=800,xlabel="",ylabel="",label="",density=True):
	import rpy2.robjects as ro
	from rpy2.robjects.packages import importr
	import rpy2.robjects.lib.ggplot2 as ggplot2
	r=ro.r
	
	#x=ro.FloatVector(x)
	#y=ro.FloatVector(y)
	
	datadict={}
	datadict['x']=ro.FloatVector(x)
	datadict['y']=ro.FloatVector(y)
	dataf=ro.DataFrame(datadict)
	gp = ggplot2.ggplot(dataf)
	
	
	#r.plot(x,y,xlab=xlabel,ylab=ylabel)
	#ggplot2.stat_smooth(color='blue',size=1,method="glm",family="quasibinomial") + \
	
	if density:
		pp = gp + \
			ggplot2.aes_string(x='x', y='y') + \
			ggplot2.geom_point(size=2) + \
			ggplot2.geom_smooth(method = "lm", formula = 'y ~ poly(x, 2)', colour = "red") + \
			ggplot2.geom_density(ggplot2.aes_string(x='x',y='..scaled..')) + \
			ggplot2.opts(**{'title' : fn.split("/")[-1], 'axis.text.x': ggplot2.theme_text(size=16), 'axis.text.y': ggplot2.theme_text(size=16,hjust=1)} ) + \
			ggplot2.scale_colour_brewer(palette="Set1") 
	else:
		pp = gp + \
			ggplot2.aes_string(x='x', y='y') + \
			ggplot2.geom_point(size=2) + \
			ggplot2.geom_smooth(method = "lm", formula = 'y ~ poly(x, 2)', colour = "red") + \
			ggplot2.opts(**{'title' : fn.split("/")[-1], 'axis.text.x': ggplot2.theme_text(size=16), 'axis.text.y': ggplot2.theme_text(size=16,hjust=1)} ) + \
			ggplot2.scale_colour_brewer(palette="Set1")
		
	
	
	grdevices = importr('grDevices')
	grdevices.png(file=fn, width=w, height=h)
	pp.plot()
	grdevices.dev_off()
	print ">> saved: "+fn
	

def polyfits(ld,fitv,deg=3):
	dd=datadict(ld)
	
	header=["word"]+["pc"+str(i) for i in sorted(range(0,deg+1),reverse=True)]
	header+=['residuals', 'rank', 'rcond']
	lines=[]
	for k,vs in dd.items():
		if k==fitv: continue
		
		try:
			fit=polyfit(vs,dd[fitv])
		except:
			continue
		
		if not fit: continue
		try:
			residuals=x[1][0]
		except:
			residuals=[]
		lines.append([k]+[y for y in fit[0]]+[residuals,fit[2],fit[4]])
	
	return "\t".join(header)+"\n"+"\n".join("\t".join([str(x) for x in line]) for line in lines)
	
	
	

def pca2(dataframe,fn):
	import rpy2.robjects as ro
	from rpy2.robjects.packages import importr
	r = ro.r
	base     = importr('base')
	stats    = importr('stats')
	graphics = importr('graphics')
	
	pca = stats.princomp(dataframe)
	grdevices = importr('grDevices')
	ofn=fn.replace('pca.','pca.eigens.')
	strfacts=str(dataframe.nrow)+" items using "+str(dataframe.ncol)+" features ["+ofn.split("/")[-1]+"]"
	grdevices.png(file=ofn, width=1280, height=1200)
	graphics.plot(pca, main = "Eigenvalues for "+strfacts)
	grdevices.dev_off()
	print ">> saved: "+ofn	

	grdevices = importr('grDevices')
	ofn=fn.replace('pca.','pca.biplot.')
	strfacts=str(dataframe.nrow)+" items using "+str(dataframe.ncol)+" features ["+ofn.split("/")[-1]+"]"
	grdevices.png(file=ofn, width=1280, height=1200)
	stats.biplot(pca, scale=1,main = "biplot of "+strfacts)
	grdevices.dev_off()
	print ">> saved: "+ofn

def pca(fn,df,col=None,w=1200,h=1200):
	import rpy2.robjects as ro
	from rpy2.robjects.packages import importr
	r = ro.r
	base     = importr('base')
	stats    = importr('stats')
	graphics = importr('graphics')
	
	
	if col:
		df,factors=dataframe(df,factorcol=col)
	
	pca = stats.princomp(df)
	
	grdevices = importr('grDevices')
	ofn=fn.replace('pca.','pca.eigens.')
	strfacts=str(df.nrow)+" items using "+str(df.ncol)+" features ["+ofn.split("/")[-1]+"]"
	grdevices.png(file=ofn, width=w, height=h)
	graphics.plot(pca, main = "Eigenvalues for "+strfacts)
	if col:
		graphics.hilight(pca,factors)
	grdevices.dev_off()
	print ">> saved: "+ofn	

	grdevices = importr('grDevices')
	ofn=fn.replace('pca.','pca.biplot.')
	strfacts=str(df.nrow)+" items using "+str(df.ncol)+" features ["+ofn.split("/")[-1]+"]"
	grdevices.png(file=ofn, width=w, height=h)
	stats.biplot(pca, scale=1,main = "biplot of "+strfacts)
	grdevices.dev_off()
	print ">> saved: "+ofn



def pca0(datadict,fn):
	import rpy2.robjects as ro
	from rpy2.robjects.packages import importr
	r = ro.r
	base     = importr('base')
	stats    = importr('stats')
	graphics = importr('graphics')

	
	#for k,v in datadict.items():
	#	datadict[k]=ro.FloatVector(v)
	#dataf=ro.DataFrame(datadict)
	
	nrow,ncol=getdimensions(datadict)
	cols=sorted(datadict.keys())
	#rows=sorted(datadict[cols[0]].keys())

	
	m = base.matrix(ro.NA_Real, nrow=nrow, ncol=ncol)
	for row_i in xrange(1, nrow+1):
	    for col_i in xrange(1, ncol+1):
			#print datadict[cols[col_i-1]][row_i-1]
			#m.rx[row_i][col_i]=datadict[cols[col_i-1]][row_i-1]
			key=ro.rlc.TaggedList((row_i,col_i))
			#print key
			m.rx[key] = datadict[cols[col_i-1]][row_i-1]
	
	#print m
	
	pca = stats.princomp(m)
	#print pca[3]
	
	
	
	
	grdevices = importr('grDevices')
	ofn=fn.replace('pca.','pca.eigens.')
	strfacts=str(nrow)+" items using "+str(ncol)+" features ["+ofn.split("/")[-1]+"]"
	grdevices.png(file=ofn, width=1280, height=768)
	graphics.plot(pca, main = "Eigenvalues for "+strfacts)
	grdevices.dev_off()
	print ">> saved: "+ofn	

	grdevices = importr('grDevices')
	ofn=fn.replace('pca.','pca.biplot.')
	strfacts=str(nrow)+" items using "+str(ncol)+" features ["+ofn.split("/")[-1]+"]"
	grdevices.png(file=ofn, width=1280, height=768)
	stats.biplot(pca, scale=1,main = "biplot of "+strfacts)
	grdevices.dev_off()
	print ">> saved: "+ofn
	
def tdw(occurrences,doclengths,op='sub',threshold=5):
	tdr=[]
	corpuslength=sum(doclengths)
	sumoccurrences=sum(occurrences)
	if sumoccurrences<threshold: return 0
	for oi,ox in enumerate(occurrences):
		ex=((doclengths[oi]/corpuslength) * sumoccurrences)
		if op=='div':
			try:
				tdx=ox / ex
			except ZeroDivisionError:
				tdx=0
		else:
			tdx=ox - ex
		tdr+=[tdx]
	return tdr


def plot(ld,fn,x='year',y='tf',sum2=True):
	fn=fn.replace(" ","")
	
	qdict={}
	qdict['x']=[]
	qdict['y']=[]
	
	if sum2:
		sqdict={}
		sqdict['x']=[]
		sqdict['y']=[]
		summ=0
	for d in ld:
		qdict['x'].append(d[x])
		qdict['y'].append(d[y])

		if sum2:
			summ+=d[y]
			#print summ
			sqdict['x'].append(d[x])
			sqdict['y'].append(summ)
			

	print ">> plotting: "+fn
	print "\t\tX:["+x+"]\t"+str(len(qdict['x']))
	print "\t\tY:["+y+"]\t"+str(len(qdict['y']))

	# normalize
	#for yn in range(len(sqdict['y'])):
	#	yval=sqdict['y'][yn]
	#	sqdict['y'][yn]=yval/max(sqdict['x'])
		

	scatterplot(qdict,fn,x='x',y='y')
	if sum2:
		scatterplot(sqdict,fn.replace('.png','.sum.png'),x='x',y='y')

def plotframe(fn,df,x='x',y='y',col=None,group=None,w=1100,h=800,size=2,smooth=True,point=True,jitter=False,boxplot=False,boxplot2=False,title=False,flip=False,se=False,density=False,line=False):
	#import math, datetime
	import rpy2.robjects.lib.ggplot2 as ggplot2
	import rpy2.robjects as ro
	from rpy2.robjects.packages import importr
	grdevices = importr('grDevices')
	
	if not title:
		title=fn.split("/")[-1]

	grdevices.png(file=fn, width=w, height=h)
	gp = ggplot2.ggplot(df)
	pp = gp	
	if col and group:
		pp+=ggplot2.aes_string(x=x, y=y,col=col,group=group)
	elif col:
		pp+=ggplot2.aes_string(x=x, y=y,col=col)
	elif group:
		pp+=ggplot2.aes_string(x=x, y=y,group=group)
	else:
		pp+=ggplot2.aes_string(x=x, y=y)	

	if boxplot:
		if col:
			pp+=ggplot2.geom_boxplot(ggplot2.aes_string(fill=col),color='blue')
		else:
			pp+=ggplot2.geom_boxplot(color='blue')	
	
	if point:
		if jitter:
			if col:
				pp+=ggplot2.geom_point(ggplot2.aes_string(fill=col,col=col),size=size,position='jitter')
			else:
				pp+=ggplot2.geom_point(size=size,position='jitter')
		else:
			if col:
				pp+=ggplot2.geom_point(ggplot2.aes_string(fill=col,col=col),size=size)
			else:
				pp+=ggplot2.geom_point(size=size)


	if boxplot2:
		if col:
			pp+=ggplot2.geom_boxplot(ggplot2.aes_string(fill=col),color='blue',outlier_colour="NA")
		else:
			pp+=ggplot2.geom_boxplot(color='blue')
	
	if smooth:
		if smooth=='lm':
			if col:
				pp+=ggplot2.stat_smooth(ggplot2.aes_string(col=col),size=1,method='lm',se=se)
			else:
				pp+=ggplot2.stat_smooth(col='blue',size=1,method='lm',se=se)
		else:
			if col:
				pp+=ggplot2.stat_smooth(ggplot2.aes_string(col=col),size=1,se=se)
			else:
				pp+=ggplot2.stat_smooth(col='blue',size=1,se=se)
	
	if density:
		pp+=ggplot2.geom_density(ggplot2.aes_string(x=x,y='..count..'))
	
	if line:
		pp+=ggplot2.geom_line(position='jitter')
	
	
	pp+=ggplot2.opts(**{'title' : title, 'axis.text.x': ggplot2.theme_text(size=24), 'axis.text.y': ggplot2.theme_text(size=24,hjust=1)} )
	#pp+=ggplot2.scale_colour_brewer(palette="Set1")
	pp+=ggplot2.scale_colour_hue()
	if flip:
		pp+=ggplot2.coord_flip()



	pp.plot()
	grdevices.dev_off()
	print ">> saved: "+fn


def plot3d(fn,df,x='x',y='y',z='z',title=False,w=800,h=800):
	#import math, datetime
	import rpy2.robjects.lib.ggplot2 as ggplot2
	import rpy2.robjects as ro
	from rpy2.robjects.packages import importr
	grdevices = importr('grDevices')

	if not title:
		title=fn.split("/")[-1]

	grdevices.png(file=fn, width=w, height=h)
	
	r = ro.r
	#df=dataframe(ld)
	#frmla=ykey+" ~ "+"+".join(xkeys)
	#lm=r['glm'](formula=frmla,data=df,family='quasibinomial']
	#return "\n\n".join(str(x) for x in [ r['anova'](lm,test='Chisq') ] )
	
	r('library(scatterplot3d)')
	r['attach'](df)
	r['print'](x)
	s3d=r['scatterplot3d'](x,y,z,type="h",main=title)
	fit=r['lm']('x ~ y+z') 
	s3d.plane3d(fit)

	grdevices.dev_off()
	print ">> saved: "+fn





def scatterplot(datadict,fn,x='year',y='tf',title=''):
	import math, datetime
	import rpy2.robjects.lib.ggplot2 as ggplot2
	import rpy2.robjects as ro
	from rpy2.robjects.packages import importr
	grdevices = importr('grDevices')
	
	if not title:
		title=fn.split("/")[-1]
	

	grdevices.png(file=fn, width=1200, height=800)
	
	for k in [x,y]:
		datadict[k]=ro.FloatVector(datadict[k])
	
	dataf=ro.DataFrame(datadict)
	
	gp = ggplot2.ggplot(dataf)

	pp = gp + \
	     ggplot2.aes_string(x=x, y=y) + \
	     ggplot2.geom_point(size=2) + \
		 ggplot2.stat_smooth(color='blue',size=1,method='lm') + \
		 ggplot2.opts(**{'title' : title, 'axis.text.x': ggplot2.theme_text(size=16), 'axis.text.y': ggplot2.theme_text(size=16,hjust=1), 'background.fill': 'white'} ) + \
		 ggplot2.scale_colour_hue()

	pp.plot()
 	#ggplot2.scale_colour_brewer(palette="Set1")
	
	print ">> saved: "+fn
	# plotting code here
	#  + \
	# ggplot2.stat_quantile()
	#
	
	grdevices.dev_off()

def commafy(s, sep=','):  
    if len(s) <= 3: return s  
    return commafy(s[:-3], sep) + sep + s[-3:]

def minordmag(number):

	c=str(Decimal(str(number))).lower().strip()
	
	if "e" in c:
		
		return int(c[c.index("e")+2:])
		# 
		# print x
		# exit()
		# 
		# while x.startswith('0'):
		# 	x=x[1:]
		# return int(x)
	else:
		minordmag=1
		for n in range(c.index('.')+1,len(c)):
			if c[n]=="0":
				minordmag+=1
			else:
				break
		return minordmag
		
		
def linreg(X, Y):
	from math import sqrt
	from numpy import nan, isnan
	from numpy import array, mean, std, random
	
	if len(X)<2 or len(Y)<2:
		return 0,0,0
	"""
	Summary
	    Linear regression of y = ax + b
	Usage
	    real, real, real = linreg(list, list)
	Returns coefficients to the regression line "y=ax+b" from x[] and y[], and R^2 Value
	"""
	if len(X) != len(Y):  raise ValueError, 'unequal length'
	N = len(X)
	Sx = Sy = Sxx = Syy = Sxy = 0.0
	for x, y in map(None, X, Y):
	    Sx = Sx + x
	    Sy = Sy + y
	    Sxx = Sxx + x*x
	    Syy = Syy + y*y
	    Sxy = Sxy + x*y
	det = Sxx * N - Sx * Sx
	a, b = (Sxy * N - Sy * Sx)/det, (Sxx * Sy - Sx * Sxy)/det
	meanerror = residual = 0.0
	for x, y in map(None, X, Y):
	    meanerror = meanerror + (y - Sy/N)**2
	    residual = residual + (y - a * x - b)**2
	    
	RR = 1 - residual/meanerror if meanerror else 1
	ss = residual / (N-2) if (N-2) else 0
	Var_a, Var_b = ss * N / det, ss * Sxx / det
	#print "y=ax+b"
	#print "N= %d" % N
	#print "a= %g \\pm t_{%d;\\alpha/2} %g" % (a, N-2, sqrt(Var_a))
	#print "b= %g \\pm t_{%d;\\alpha/2} %g" % (b, N-2, sqrt(Var_b))
	#print "R^2= %g" % RR
	#print "s^2= %g" % ss
	return a, b, RR


def fitall(x,y,returnType='d'):
	import math
	logx=[math.log10(ix) if ix else 0 for ix in x]
	logy=[math.log10(iy) if iy else 0 for iy in y]
	
	## regressions
	fits={}
	fits['fit_linear']=linreg(x,y)[-1]
	fits['fit_log']=linreg(logx,y)[-1]
	fits['fit_exponential']=linreg(x,logy)[-1]
	fits['fit_power']=linreg(logx,logy)[-1]
	
	if returnType=='d':
		return fits
	else:
		"""o=[]
		for k,v in sorted(fits.items()):
			l='Regression of type {0} has R^2 = {1}'.format(k.upper(), v)
			o+=[l]
		return "\n".join(o)"""
		maxrr=max(fits.values())
		for k,v in fits.items():
			if not v==maxrr: continue
			return 'Regression of type {0} has R^2 = {1}'.format(k.upper(), v)

	return fits
	


def product(*args):
	if not args:
		return iter(((),)) # yield tuple()
	return (items + (item,) 
		for items in product(*args[:-1]) for item in args[-1])

def fitexp(values):
	ranks=[]
	vals=[]
	import math
	for rank,val in enumerate(sorted(values, reverse=True)):
		ranks+=[math.log(rank+1)]	
		vals+=[math.log(val)]
	
	return (correlate(ranks,vals), linreg(ranks,vals))

def gcd(a, b):
    """Return greatest common divisor using Euclid's Algorithm."""
    while b:      
        a, b = b, a % b
    return a

def lcm(a, b):
    """Return lowest common multiple."""
    return a * b // gcd(a, b)

def lcmm(*args):
    """Return lcm of args."""   
    return reduce(lcm, args)

	
def pair(*args):
	z=[]
	ps=product(*args)
	for p in ps:
		try:
			x,y=p
		except:
			continue
		if x==y: continue
		if x!=sorted([x,y])[0]: continue
		z.append((x,y))
	return z

def value_sequence(dict):
	return [v for k,v in sorted(dict.items())]

def corrwithsum(tf_tuple):
	fieldtfs=tf_tuple[0]
	wordtfs=tf_tuple[1]
	corrs={}
	for word,tfdict in wordtfs.items():
		xtfdict={}
		for k,v in fieldtfs.items():
			try:
				xtfdict[k]=fieldtfs[k]-tfdict[k]
			except KeyError:
				xtfdict[k]=fieldtfs[k]
		try:
			corrs[word]=correlate_dictionaries(xtfdict,tfdict)
		except ZeroDivisionError:
			corrs[word]=[0,0]
	return corrs
		
