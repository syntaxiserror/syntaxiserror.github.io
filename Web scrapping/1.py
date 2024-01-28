from bs4 import BeautifulSoup
import requests

html = requests.get("http://quotes.toscrape.com/").text

soup = BeautifulSoup(html, 'lxml')


quotes = soup.find_all("div", class_="quote")
for index, quote in enumerate(quotes):
    author = quote.find("small", class_="author").text
    link = "http://quotes.toscrape.com" + quote.a["href"]

    with open (f"E:/python projects/WEB/Scraping/logs/{index}.txt", "w") as file:
        file.write(f"Author: {author}")
        file.write(" ")
        file.write(f"Link: {link}")
    print(f"File {index} is saved")