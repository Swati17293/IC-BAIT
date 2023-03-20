import csv
import os
import json
import time


f = open('headline_bias.csv')

f1 = open('headline_bias_pro.csv')

processed_headlines = []

for lines in f1:
    processed_headlines.append(lines.split(',')[2])

csvfile  = open("Knowledge.csv", "a", newline='\n') 
csvwriter = csv.writer(csvfile, delimiter='\t')

line_cnt = 0
for lines in f:
    
    x = lines.split(',')
    bias = x[0]
    headline = x[1].replace('\n','')
    headline_pro = processed_headlines[line_cnt].replace('\n','')
    line_cnt += 1
    headline_curl = headline.replace(' ','+').replace('\n','')

    success = False
    num_try = 0
    while success is False:

        if num_try >= 50: 
            pass

        try:
            
            curl_query = 'curl  --header "Content-Type: application/json" "https://comet-atomic-2020.apps.allenai.org/comet?head=' + headline_curl + '&rel=xAttr&rel=xNeed&rel=xIntent&rel=xEffect&rel=xWant&rel=xReact&rel=oWant&rel=oEffect&rel=oReact&decode_method=beam"'

            curl_output = os.popen(curl_query).read()

            res = json.loads(curl_output)
            res = res['answer']
            time.sleep(10) 

            kg = []

            for i in range(0,len(res)):
                print(res[i])

                x = res[i]

                for z in range(0,5):
                    if sum(c.isalpha() for c in x[z]) > 1 and x[z].strip() != 'none':
                        y = x[z]
                        break
                    y = 'none'

                kg.append(y)

            print('\n')
            print(kg)
            print('\n')

            flag = 0

            for i in range(0,len(res)):
                if kg[i] != ' none':
                    flag = 1
                    break

            IC_Knwl = ''

            if flag == 1:
                IC_Knwl = "PersonX is" + kg[0] + ", needed" + kg[1] + ", intended" + kg[2] + "," + kg[3] + ", wants" + kg[4] + ", feels" + kg[5] + ". Others want" + kg[6] + "," + kg[7] + ", feel" + kg[8] + "." 
            else: 
                IC_Knwl = 'none'
        
            print('\n')
            print(headline)
            print(IC_Knwl)

            data_write = []

            data_write.append(bias)
            data_write.append(headline)
            data_write.append(headline_pro)
            data_write.append(IC_Knwl)

            csvwriter.writerow(data_write)

            success = True

        except:
            num_try += 1
            continue

    