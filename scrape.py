import requests
from bs4 import BeautifulSoup
from lxml import html

def writeLink(link):
    response = requests.get(link)
    soup = BeautifulSoup(response.content, 'html.parser')

    date = soup.find('div', {'class': 'field-docs-start-date-time'}).get_text()
    dateMod = date.strip().replace(', ', '_').replace(' ', '_')
    fname = f"data/transcripts/{dateMod}.txt"
    mname = f"data/moderators/{dateMod}.txt"
    pname = f"data/participants/{dateMod}.txt"

    with open(fname, 'w') as f, open(mname, 'w') as m, open(pname, 'w') as p:
        transcript = soup.find('div', {'class': 'field-docs-content'})
        ps = transcript.findAll('p')
        for i in range(len(ps)):
            if i == 0: # Participants
                p.write(ps[i].get_text())
            if i == 1: # Moderators
                m.write(ps[i].get_text())
            if i > 1: # Debate Content
                f.write(f'{ps[i].get_text()}\n')


def main():
    url = 'https://www.presidency.ucsb.edu/documents/presidential-documents-archive-guidebook/presidential-campaigns-debates-and-endorsements-0'

    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    links = []
    possibleLinks = soup.find_all("a") # Find all elements with the tag <a>
    for link in possibleLinks:
        l = link.get("href")
        if l and len(l) >= 42 and l[:42] == "https://www.presidency.ucsb.edu/documents/": # could prob clean up w regex
            links.append(l)

    for link in links:
        writeLink(link)
    
    
if __name__ == '__main__':
    main()