import pandas as pd
import urllib
from datetime import datetime, timedelta
import urllib.request

# https://cwcgom.aoml.noaa.gov/erddap/griddap/noaa_aoml_seascapes_8day.csv?CLASS%5B(2012-01-01T12:00:00Z)%5D%5B(-47.975):(-38.975)%5D%5B(-67.975):(-50.975)%5D&.draw=surface&.vars=longitude%7Clatitude%7CCLASS&.colorBar=%7C%7C%7C%7C%7C&.bgColor=0xffccccff

# CLASS
current_date = datetime.strptime("2012-01-01", "%Y-%m-%d")
for i in range(230):
    print(f"CLASS: {current_date}")
    inline_date = current_date.strftime("%Y-%m-%d")
    url = f"https://cwcgom.aoml.noaa.gov/erddap/griddap/noaa_aoml_seascapes_8day.csv?CLASS%5B({inline_date}T12:00:00Z)%5D%5B(-47.975):(-38.975)%5D%5B(-67.975):(-50.975)%5D&.draw=surface&.vars=longitude%7Clatitude%7CCLASS&.colorBar=%7C%7C%7C%7C%7C&.bgColor=0xffccccff"    
    urllib.request.urlretrieve(url, filename = f"data/seascapes/class/noaa_seascape_CLASS_Patagonia_Shelf_{inline_date}.csv")    
    next_date = current_date + timedelta(days=8)
    current_date = next_date
    
# https://cwcgom.aoml.noaa.gov/erddap/griddap/noaa_aoml_seascapes_8day.csv?P%5B(2012-01-01T12:00:00Z)%5D%5B(-47.975):(-38.975)%5D%5B(-67.975):(-50.975)%5D&.draw=surface&.vars=longitude%7Clatitude%7CP&.colorBar=%7C%7C%7C%7C%7C&.bgColor=0xffccccff
# PROB
current_date = datetime.strptime("2012-01-01", "%Y-%m-%d")
for i in range(230):
    print(f"PROB: {current_date}")
    inline_date = current_date.strftime("%Y-%m-%d")
    url = f"https://cwcgom.aoml.noaa.gov/erddap/griddap/noaa_aoml_seascapes_8day.csv?P%5B({inline_date}T12:00:00Z)%5D%5B(-47.975):(-38.975)%5D%5B(-67.975):(-50.975)%5D&.draw=surface&.vars=longitude%7Clatitude%7CP&.colorBar=%7C%7C%7C%7C%7C&.bgColor=0xffccccff"    
    urllib.request.urlretrieve(url, filename = f"data/seascapes/prob/noaa_seascape_PROB_Patagonia_Shelf_{inline_date}.csv")    
    next_date = current_date + timedelta(days=8)
    current_date = next_date

