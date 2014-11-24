template={'File name*': '',
'Photo title (max 50 characters)*': '',
'Content': '',
'Keywords (max 500 characters)': '',
'Date (See instructions for format)': '',
'Interval (Years | 1, 2, 3, 5, 10, 15)': '',
'Latitude , Longitude': '',
'Story': '',
'Use Date (TRUE or FALSE)': '',
'Date Range? (Y or N)': '',
'Licence (See instructions for options)': '',
'Copyright holder': '',
'Author': '',
'Original Link': '',
'Repository or Archive': '',
'Notes': ''}

N=30


fn='results_bypassage3.xlsx'
ld=pytxt.xlsx2ld(fn)
print len(ld)
ld=[d for d in ld if d['lat'] and d['long']]
print len(ld)