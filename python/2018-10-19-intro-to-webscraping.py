from bs4 import BeautifulSoup
import requests

url = "https://www.phivolcs.dost.gov.ph/html/update_SOEPD/EQLatest-Monthly/2018/2018_September.html"
res = requests.get(url)

html = BeautifulSoup(res.content, "lxml")
# print(html.children)
qres = html.select(".MsoNormalTable")
# qres[3]
print(len(list(qres[2].children)))

print(type(html))


def htmldoc(url: str):
    return BeautifulSoup(requests.get(url).content, "html.parser")

def scraper(html: BeautifulSoup):
    date, locn = [], []
    latd, lond, dept, magn = [], [], [], []

    tbody = html