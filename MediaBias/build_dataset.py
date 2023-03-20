from bs4 import BeautifulSoup
import requests
import csv
import json
import os

if os.path.exists('dic_headline.json'):
    dic = json.load(open("dic_headline.json"))
else:
    dic = {}

# pagenum = 0-128 #08-Jan-2022
for pagenum in range(0,128):

    try:

        print('pagenum:' + str(pagenum))

        page = requests.get("https://www.allsides.com/story/admin?page="+str(pagenum))

        soup = BeautifulSoup(page.content, 'html.parser')

        page_links = soup.find_all(class_="views-field views-field-name")

        max_links = len(page_links) + 1

        for num in range(1,max_links):

            print(num)

            try:
                
                page_link = page_links[num].find('a', href=True)
                page_link = 'https://www.allsides.com' + page_link['href']

                data_page = requests.get(page_link)
                data_soup = BeautifulSoup(data_page.content, 'html.parser')

                data_page_links = data_soup.find_all(class_='news-title')
                class_page_links = data_soup.find_all(class_='bias-image')

                len_data_page_links = len(data_page_links)

                for i in range(0,len_data_page_links):

                    try:

                        headline = data_page_links[i].find('a').get_text()

                        if headline.endswith('...'):

                            headline_link = data_page_links[i].find('a')['href']

                            head_page = requests.get(headline_link)
                            head_soup = BeautifulSoup(head_page.content, 'html.parser')
                            headline = head_soup.find("title").get_text().split('|')[0].strip()

                        bias = ((class_page_links[i].find('img')['title']).split(':'))[1].strip()

                        dic[headline] = bias
                        
                    except:
                        pass
            except:
                pass
    except:
        print('repeat')

json.dump(dic,open("dic_headline.json",'w'))

csvfile  = open("headline_bias.csv", "w", newline='\n') 
csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"')

for key in dic:
    data = [dic[key]]
    data.append(key)
    csvwriter.writerow(data)

csvfile.close()