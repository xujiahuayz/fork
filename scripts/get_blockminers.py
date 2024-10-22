import requests
import pandas as pd


url = [
    "https://explorer.api.cloverpool.com/chain/block/list?app_a=S2Q4QUljMHEwNzNoMAI5XdXXMAgwqxia4P8VScnvddXEG5EguXvUEXk4ccTC&app_b=Kd8AIc0q073h0&nonce=DJI22151IXIC7512&timestamp=1729584638&coins=btc&page=4&page_size=1&start_time=1726959600&end_time=1729637999&is_page_by_day=true&sign=l4rYPV75vm5Mgs4LAQKGh9mQ8DY5dYam3JSzZZpwVPw=",
    "https://explorer.api.cloverpool.com/chain/block/list?app_a=S2Q4QUljMHEwNzNoMAI5XdXXMAgwqxia4P8VScnvddXEG5EguXvUEXk4ccTC&app_b=Kd8AIc0q073h0&nonce=DJI22151IXIC7512&timestamp=1729584654&coins=btc&page=5&page_size=1&start_time=1726959600&end_time=1729637999&is_page_by_day=true&sign=Wu54XefatFPdCPKctLp0HdoUXu/+DA0Kw6kICr7WEfw=",
]

full_list = []
for i in range(2):

    w = url[i]
    response = requests.get(w)
    this_list = response.json()["data"]["btc"]["list"]
    full_list.extend(this_list)

block_miners_df = pd.DataFrame(full_list)
