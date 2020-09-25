#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 17:59:41 2020

@author: EJ Gustavo tha poet
"""


import random
import re
import sys
import time
from lxml import html
import bs4
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


def delay() -> None:
    time.sleep(random.uniform(15, 30))
    return None


base: str = "https://www.sequoiacap.com/companies/"
url = "https://www.premierleague.com/players/"
stats = "stats"
overview = "overview"


delay()
r: requests.Response = requests.get(url)
soup: bs4.BeautifulSoup = BeautifulSoup(r.content, "html.parser")
print(soup.prettify())

"""

for player in soup.find_all(
        "a", {"class": "playerName"}):
    
    link = player.get('href')
    #replace overview with stats in the link collected from premierleague.com
    #result = re.sub(r"overview", stats, link)
    link = link.replace("overview", stats)

r = requests.get("https://www.premierleague.com" + link)
detailed_soup = BeautifulSoup(r.content, "html.parser")
#print(detailed_soup)


page = requests.get('https://www.premierleague.com/players')
tree = html.fromstring(page.content)

#Using the page's CSS classes, extract all links pointing to a team
linkLocation = tree.cssselect('.indexItem')

#Create an empty list for us to send each team's link to
teamLinks = []

#For each link...
for i in range(0,583):
    
    #...Find the page the link is going to...
    temp = linkLocation[i].attrib['href']
    
    #...Add the link to the website domain...
    temp = "http://www.premierleague.com/" + temp
    
    #...Change the link text so that it points to the squad list, not the page overview...
    temp = temp.replace("overview", stats)
    
    #...Add the finished link to our teamLinks list...
    teamLinks.append(temp)

    print(temp)
    """