import csv

csvfile  = open("headline_bias.csv", "w", newline='\n') 
csvwriter = csv.writer(csvfile, delimiter=',')


f = open('goodnews.tsv')

for lines in f:

    line = lines.split('\t')

    bias_score = line[1].strip()
    headline = line[0]
    bias = ''
    
    if bias_score:

        bias_score = int(bias_score)

        if bias_score in range(-6,7):
            bias = 'Center'

        if bias_score in range(18,31):
            bias = 'Right'

        if bias_score in range(-30,-17):
            bias = 'Left'

        if bias != '':

            headline = headline.replace(',','')
            headline = headline.replace('"','')
            data = []   
            data.append(bias)
            data.append(headline)

            csvwriter.writerow(data)

csvfile.close()
    
    