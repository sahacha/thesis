from bs4 import BeautifulSoup
import requests
import bs4

# url = "https://www.chiangmaicitylife.com/news/chiang-mai-news/"
# response = requests.get(url)
# soup = BeautifulSoup(response.text, 'html.parser')
# print(soup.prettify())

r = requests.get('https://www.wongnai.com/restaurants/sushimasa')
print(r.text)
