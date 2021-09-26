from bs4 import BeautifulSoup
import requests

BASE_URL = 'https://en.wikipedia.org/wiki/'


def extract_wiki(page_name: str):
    url = BASE_URL + page_name.replace(' ', '_')

    raw_text = requests.get(url).text
    soup = BeautifulSoup(raw_text, 'html.parser')

    text_blocks = soup.find('div', class_='mw-parser-output').find_all('p')
    content = ''
    for text_block in text_blocks:
        for string in text_block.stripped_strings:
            content += string + ' '

    return content, url
