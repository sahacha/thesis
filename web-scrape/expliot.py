from bs4 import BeautifulSoup
import requests

url = "https://www.chiangmaicitylife.com/news/chiang-mai-news/"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
print(soup.prettify())