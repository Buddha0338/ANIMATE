import requests
import pandas as np
import os

def fetch_gbif_data(dataset_key, limit=100, offset=0, country=None, output_file='../data/gbif_inaturalist_data.csv'):
    url = "https://api.gbif.org/v1/occurrence/search"
    params = {
        "datasetKey": dataset_key,
        "limit": limit,
        "offset": offset,
    }
    if country:
        params["country"] = country

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        occurrences = data['results']
        df = pd.DataFrame(occurrences)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
    else:
        print(f"Failed to fetch data: {response.status_code}")

if __name__ == "__main__":
    # Replace with your specific dataset key from iNaturalist
    DATASET_KEY = "your-dataset-key"
    LIMIT = 100  # Number of records to fetch
    OFFSET = 0   # Starting point of the records
    COUNTRY = "US"  # Example parameter, you can modify as needed

    fetch_gbif_data(DATASET_KEY, LIMIT, OFFSET, COUNTRY)
