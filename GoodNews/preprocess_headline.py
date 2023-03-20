import re
import csv

#case insensitive replace
def ireplace(old, new, text):
    idx = 0
    while idx < len(text):
        index_l = text.lower().find(old.lower(), idx)
        if index_l == -1:
            return text
        text = text[:index_l] + new + text[index_l + len(old):]
        idx = index_l + len(new) 
    return text

contractions = {
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"i'd": "I would",
"i'd've": "I would have",
"i'll": "I will",
"i'll've": "I will have",
"i'm": "I am",
"i've": "I have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "he would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}

extra = ['Fact check: ','Republicans: ',': report','Poll: ','Exclusive: ','Media Bias Alert: ','BREAKING: ','democrats: ', ' (+video)',
         ': Attorney','The Latest: ',': poll','OPINION: ','Reports: ','Official: ','CNN: ','Analysis: ','Pence: ',': New York Times',
         'Video: ','Watch: ','experts: ','. . .','...','!','@','#','$','*','(',')','+','|',',','?']
         
abbrv_dic = {
         'jan.' : 'january',
         'feb.' : 'february',
         'mar.' : 'march',
         'apr.' : 'april',
         'may.' : 'may',
         'jun.' : 'june',
         'jul.' : 'july',
         'aug.' : 'august',
         'sep.' : 'september',
         'sept.' : 'september',
         'oct.' : 'october',
         'nov.' : 'november',
         'dec.' : 'december',
         '&' : 'and'
        }

# --------------------------------------------------------

f = open('headline_bias.csv')

csvfile  = open("headline_bias_pro.csv", "w", newline='\n') 
csvwriter = csv.writer(csvfile, delimiter=',')


for lines in f:

    line = lines.strip().replace('"','').replace('“','').replace('”','').split(',')
    headline = ' '.join(line[1:]).replace(',','').replace('  ',' ')
    headline_old = headline
    bias = line[0]

    headline = headline.encode("ascii", "ignore")
    headline = str(headline, 'utf-8', 'ignore')

    headline = re.sub(r'(\d)\s+(\d)', r'\1\2', headline)
    headline = re.sub(r'(\d)\,(\d)', r'\1\2', headline)
    headline = headline.replace('%',' percent')

    headline_ = headline.split()

    for n in headline_:
        try:
            headline = headline.replace(n,str(round(float(n))))
        except:
            pass

    for word in extra:
        headline = headline.replace(word, '')

    for word in abbrv_dic:
        wrd_old = ' ' + word + ' '
        wrd_new = ' ' + abbrv_dic[word] + ' '
        headline = ireplace(wrd_old, wrd_new, headline)

    for word in abbrv_dic:
        wrd_old = ' ' + word
        wrd_new = ' ' + abbrv_dic[word]
        headline = ireplace(wrd_old, wrd_new, headline)

    for word in headline.split():
        if word.lower() in contractions:
            headline = headline.replace(word, contractions[word.lower()])
        
        if re.search(r"'$", word) is not None:
            word_new = re.sub(r"'$", '', word)
            headline = headline.replace(word, word_new)
            word = word_new

        if re.search(r"'[\W]$", word) is not None:
            word_new = re.sub(r"'[\W]$", '', word)
            headline = headline.replace(word, word_new)
            word = word_new

        if re.search(r"^'", word) is not None:
            word_new = re.sub(r"^'", '', word)
            headline = headline.replace(word, word_new)
            word = word_new

    headline = headline.replace('.','')
    data = []   
    data.append(bias)
    data.append(headline_old)
    data.append(headline.lower())

    csvwriter.writerow(data)

csvfile.close()