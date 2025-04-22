import csv
import json

csv_file = '/Users/aneesh/Downloads/airports.csv'      # Replace with your actual file name
json_file = 'cities_iata.json'

city_iata = {}

with open(csv_file, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        city = row['City'].strip()
        iata = row['IATA'].strip()
        if city and iata:
            city_iata[city] = iata

with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(city_iata, f, indent=4)

print(f"Saved {len(city_iata)} city-IATA pairs to {json_file}")
