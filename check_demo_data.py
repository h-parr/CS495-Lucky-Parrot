import pandas as pd
import numpy as np

df = pd.read_csv('data/demo_trips_subset.csv')
print(f'Total rows: {len(df)}')
print(f'Unique trips: {df["trip_id"].nunique()}')
print("\nTrip statistics:")
for trip_id in sorted(df['trip_id'].unique()):
    trip = df[df['trip_id'] == trip_id]
    lat_range = trip['Latitude'].max() - trip['Latitude'].min()
    lon_range = trip['Longitude'].max() - trip['Longitude'].min()
    speed_range = trip['Speed'].max() - trip['Speed'].min()
    weight_min = trip['Weight_lbs'].min()
    weight_max = trip['Weight_lbs'].max()
    print(f"{trip_id}: {len(trip)} pings | lat_range={lat_range:.3f} lon_range={lon_range:.3f} speed_range={speed_range:.1f} weight={weight_min:.0f}-{weight_max:.0f}")
