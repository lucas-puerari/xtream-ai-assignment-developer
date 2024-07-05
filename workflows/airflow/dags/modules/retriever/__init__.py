import requests


def download_data(url):
    response = requests.get(url)
    response.raise_for_status()
    content = response.content.decode('utf-8')
    return content
