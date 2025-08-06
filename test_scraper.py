from api.scraper import scrape_data

question = "Scrape GDP data from this link: https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)"

# Add the missing closing parenthesis
question = "Scrape GDP data from this link: https://en.wikipedia.org/wiki/List_of_countries_by_GDP_%28nominal%29"

data, summary = scrape_data(question)

print(summary)
print(data[:3])
