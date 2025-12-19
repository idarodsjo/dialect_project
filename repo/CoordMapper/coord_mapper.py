from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time
from typing import Optional, Tuple


class CoordMapper:
    def __init__(self, user_agent="coord_mapper", timeout: int = 5):
        self.geolocator = Nominatim(user_agent=user_agent, timeout=timeout)
        # add delay between calls to play nice with Nominatim
        self._geocode = RateLimiter(self.geolocator.geocode, min_delay_seconds=1.0)


    
    def get_coordinates(self, place: str, area: Optional[str] = None, country: str = "Norway") -> Optional[Tuple[float, float]]:
        """
        Returns (latitude, longitude) for a given place and area/county.
        Tries with county first, then falls back to just (place, country).
        """
        # prefer county+area if provided (because of that one municipality duplicate)
        queries = []
        if place and area:
            queries.append(f"{place}, {area}, {country}")
        if place:
            queries.append(f"{place}, {country}")

        for q in queries:
            try:
                location = self._geocode(q)
                if location:
                    return (location.latitude, location.longitude)
            except (GeocoderTimedOut, GeocoderServiceError):
                time.sleep(2)   # nominatim goes crazy if there are too many requests witthout breaks
                continue
        return None


def main():
    data = """tid\tbirth\tsex\tage\tagegroup\tplace\tarea\tregion\tcountry
    aaseral_01um\t1988\tM\t19\tA\tÅseral\tVest-Agder\tSørlandet\tNorway
    aaseral_02uk\t1981\tF\t26\tA\tÅseral\tVest-Agder\tSørlandet\tNorway
    aaseral_03gm\t1933\tM\t75\tB\tÅseral\tVest-Agder\tSørlandet\tNorway
    """
    # Load into DataFrame
    df = pd.read_csv(StringIO(data), sep="\t")

    mapper = CoordMapper()

    # Get coordinates for each row
    for _, row in df.iterrows():
        print("Processing:", row['tid'])
        print(f"Place: {row['place']}, Area: {row['area']}")
        lat, lon = mapper.get_coordinates(row['place'], row['area'])
        print(f"{row['tid']} -> {row['place']}, {row['area']} -> ({lat}, {lon})")

def example_usage():
    mapper = CoordMapper()

    df = pd.read_csv("file.tsv", sep="\t")

    df["latitude"], df["longitude"] = zip(*df.apply(lambda row: mapper.get_coordinates(row["place"], row["area"]), axis=1))
    df.to_csv("file_with_coords.tsv", sep="\t", index=False)


if __name__ == "__main__":
    # Example usage
    # example_usage()
    import pandas as pd
    main()