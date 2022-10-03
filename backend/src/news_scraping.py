from bs4 import BeautifulSoup
import requests
import json



def gain_links_ap_news(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html5lib')
    links_non_filtered = [i for i in soup.find_all('a')]
    lista = []
    for element in links_non_filtered:
        if 'href="/article/' in str(element):
            index, fine = str(element).find('href="/article/'), str(element).find('">')
            if "https://apnews.com"+str(element)[index+6:fine+1] not in lista:
                lista.append("https://apnews.com"+str(element)[index+6:fine+1])
    return lista


politics_list = gain_links_ap_news("https://apnews.com/hub/politics?utm_source=apnewsnav&utm_medium=navigation")
sports_list = gain_links_ap_news("https://apnews.com/hub/sports?utm_source=apnewsnav&utm_medium=navigation")