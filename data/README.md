# Data Directory

## Structure

- `cleaned/`: Cleaned flight data ready for analysis
  - `flights_2024.parquet`: Main dataset for 2024 (YEAR=2024)
  
## Dataset Schema

The cleaned dataset contains the following columns:

- **YEAR** (int): Year of flight
- **MONTH** (int): Month (1-12)
- **FL_DATE** (date): Flight date (YYYY-MM-DD)
- **OP_UNIQUE_CARRIER** (string): Airline carrier code
- **TAIL_NUM** (string, nullable): Aircraft tail number
- **OP_CARRIER_FL_NUM** (int): Flight number
- **ORIGIN_AIRPORT_ID** (int): Origin airport ID
- **ORIGIN** (string): Origin airport IATA code
- **ORIGIN_CITY_NAME** (string): Origin city name
- **ORIGIN_STATE_NM** (string): Origin state name
- **DEST** (string): Destination airport IATA code
- **DEST_CITY_NAME** (string): Destination city name
- **DEST_STATE_NM** (string): Destination state name
- **DEP_TIME** (float/int, nullable): Departure time in HHMM format
- **DEP_DELAY** (float): Departure delay in minutes (negative = early)
- **ARR_TIME** (float/int, nullable): Arrival time in HHMM format
- **ARR_DELAY** (float): Arrival delay in minutes (negative = early)
- **CANCELLED** (float/int): Cancellation indicator (0/1)
- **AIR_TIME** (float, nullable): Airborne time in minutes
- **FLIGHTS** (float/int): Number of flights (usually 1)
- **DISTANCE** (float): Distance in miles

## Data Source

The dataset should be placed in `data/cleaned/flights_2024.parquet`.

## Validation

Run `python scripts/00_validate_inputs.py` to validate the dataset schema and constraints.
