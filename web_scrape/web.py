import requests
from bs4 import BeautifulSoup
import pandas as pd

# Fetch the main page with the list of links
url = 'https://medlineplus.gov/ency/encyclopedia_A.htm'
response = requests.get(url)

# Parse the content with BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# Find the ARTICLE section
article_section = soup.find('article')

# Find the <ul id="index"> within the ARTICLE section
ul_index = article_section.find('ul', id='index')

# Prepare a list to store the data for each page
data = []

# Extract the top 100 links within the <ul id="index"> section
if ul_index:
    links = ul_index.find_all('a')[:100]  # Limit to top 100 links
    
    for link in links:
        href = link.get('href')
        if href:
            # Handle relative URLs (convert them to absolute URLs)
            full_url = requests.compat.urljoin(url, href)
            
            # Fetch the content of the linked page
            page_response = requests.get(full_url)
            page_soup = BeautifulSoup(page_response.content, 'html.parser')
            
            # Find the page title from the <h1> tag inside <div class="page-title">
            page_title = None
            page_title_div = page_soup.find('div', class_='page-title')
            if page_title_div:
                page_title_h1 = page_title_div.find('h1', class_='with-also', itemprop='name')
                if page_title_h1:
                    page_title = page_title_h1.get_text(strip=True)
            
            # Scrape the content from the div with id="ency_summary"
            summary_content = None
            ency_summary = page_soup.find('div', id='ency_summary')
            if ency_summary:
                summary_content = ency_summary.get_text(strip=True)

            # Scrape and preserve the order of headings, paragraphs, and list items in <article><div id="d-article"><div class="main">
            main_div_content = None
            article = page_soup.find('article')
            d_article_div = article.find('div', id='d-article') if article else None
            main_div = d_article_div.find('div', class_='main') if d_article_div else None
            
            if main_div:
                # Extract all content inside <div class="main"> in the order it appears
                all_text = []
                
                for element in main_div.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']):
                    all_text.append(element.get_text(strip=True))
                
                # Join all the extracted text into a single string
                main_div_content = '/ '.join(all_text)

            # Append the data as a dictionary to the list
            data.append({
                'Title': page_title,
                'URL': full_url,
                'Summary Content': summary_content,
                'Main Content': main_div_content
            })

# Convert the data to a DataFrame for better readability
df = pd.DataFrame(data)
file_path = r"C:\Users\yufei\Programming\DeepTarget\web_scrape" ## Change the file path

# Write the DataFrame to an Excel file
df.to_excel("scraped_data.xlsx", index=False)

print("Data has been written to 'scraped_data.xlsx'")