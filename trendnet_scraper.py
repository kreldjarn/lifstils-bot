from bs4 import BeautifulSoup as bs
from urllib import request
import re, os

URL = 'http://trendnet.is'
headings_path = 'pool/new/headings.txt'
bodies_path = 'pool/new/bodies.txt'
try:
    os.remove(headings_path)
    os.remove(bodies_path)
except Exception as e:
    pass

print('Requesting URL {}'.format(URL))

r = request.urlopen(URL).read()
soup = bs(r, 'html.parser')
links = list(map(lambda l: l['href'], soup.find_all('a', {'class': ['nav-link']})[:-2]))

print('Found {} links:'.format(len(links)))
for l in links:
    print(l)


def print_link(link):
    print('Accessing {}'.format(link))

    r = request.urlopen(link).read()
    soup = bs(r, 'html.parser')
    heading_containers = soup.find_all('header');
    articles = soup.find_all('section', class_='post-entry')

    with open(headings_path, 'a') as out:
        for h in heading_containers:
            heading = h.find('h1')
            if heading:
                out.write(heading.get_text())
                out.write(' ')

    with open(bodies_path, 'a') as out:
        for a in articles:
            paragraphs = a.find_all('p')
            for p in paragraphs:
                out.write(p.get_text())
                out.write(' ')

    next_page = soup.find('a', class_='next')

    if next_page is not None:
        print_link(next_page['href'])

for l in links[3:]:
    print_link(l)
