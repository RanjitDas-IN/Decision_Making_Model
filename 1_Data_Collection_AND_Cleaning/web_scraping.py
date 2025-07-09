import requests

# url = "https://movies.fandom.com/wiki/Iron_Man/Transcript"
# url = "http://www.script-o-rama.com/movie_scripts/a2/iron-man-script-transcript.html"
# url = r"http://www.script-o-rama.com/movie_scripts/t/3-iron-script-transcript.html"
url = r"http://www.script-o-rama.com/movie_scripts/f/forrest-gump-script-transcript-hanks.html"
response = requests.get(url)

if response.status_code == 200:
    with open("1_Pipeline_Data_Acquisition/General_scraping_data.txt", "w", encoding="utf-8") as file:
    # with open("/home/ranjit/Desktop/Decision_Making_Model/1_Pipeline_Data_Acquisition/scraping.txt", "w", encoding="utf-8") as file:
        file.write(response.text)
    print("Raw HTML saved to reeeen.txt")
else:
    print(f"Failed to fetch the page. Status code: {response.status_code}")




# import requests
# from bs4 import BeautifulSoup

# # Target URL
# url = "http://www.script-o-rama.com/movie_scripts/a2/iron-man-script-transcript.html"

# # Send HTTP GET request
# response = requests.get(url)

# # Check if the request was successful
# if response.status_code == 200:
#     soup = BeautifulSoup(response.text, 'html.parser')

#     # The script content is inside the <pre> tag
#     script_tag = soup.find('pre')
#     if script_tag:
#         script_text = script_tag.get_text()

#         # Save to a file
#         with open("/home/ranjit/Desktop/Decision_Making_Model/1_Pipeline_Data_Acquisition/scraping.txt", "w", encoding="utf-8") as file:
#             file.write(script_text)
#         print("Script saved to reeeen.txt")
#     else:
#         print("Script content not found on the page.")
# else:
#     print(f"Failed to fetch page. Status code: {response.status_code}")
