# Flight Operations Dataset (2025)

This repository contains airline flight operations records for 2025 (monthly extracts merged into a single tabular dataset). Each row represents **one operated (or scheduled) flight leg** for a specific flight date, including origin/destination, carrier identifiers, timings, delays, cancellations, and distance.

## Files

- **Data file(s):**
  - `*.csv` (example structure shown below)
  - Columns are consistent across monthly extracts and can be concatenated to form a full-year dataset.

## Data Schema

**Delimiter:** comma (`,`)  
**Header:** present  
**Row granularity:** one row per flight leg per day

### Columns

| Column | Type (suggested) | Description |
|---|---:|---|
| `YEAR` | int | Calendar year of the flight date (e.g., `2025`). |
| `MONTH` | int | Month number (1–12) of the flight date. |
| `FL_DATE` | date | Flight date (local to reporting standard), formatted as `YYYY-MM-DD`. |
| `OP_UNIQUE_CARRIER` | string | Unique operating carrier code (e.g., `AA`). |
| `TAIL_NUM` | string | Aircraft tail number (registration), if available (e.g., `N101NN`). |
| `OP_CARRIER_FL_NUM` | int | Operating carrier flight number. |
| `ORIGIN_AIRPORT_ID` | int | Numeric airport identifier for origin airport. |
| `ORIGIN` | string | Origin airport IATA code (e.g., `LAX`). |
| `ORIGIN_CITY_NAME` | string | Origin city and state (e.g., `Los Angeles, CA`). |
| `ORIGIN_STATE_NM` | string | Origin state name (e.g., `California`). |
| `DEST` | string | Destination airport IATA code (e.g., `BOS`). |
| `DEST_CITY_NAME` | string | Destination city and state (e.g., `Boston, MA`). |
| `DEST_STATE_NM` | string | Destination state name (e.g., `Massachusetts`). |
| `DEP_TIME` | float/int | Actual departure time in local time as `HHMM` (e.g., `828` for 08:28). May be null/blank. |
| `DEP_DELAY` | float | Departure delay in minutes. Negative values indicate early departure. |
| `ARR_TIME` | float/int | Actual arrival time in local time as `HHMM` (e.g., `1617` for 16:17). May roll past midnight. |
| `ARR_DELAY` | float | Arrival delay in minutes. Negative values indicate early arrival. |
| `CANCELLED` | float/int | Cancellation indicator (`1` = cancelled, `0` = not cancelled). |
| `AIR_TIME` | float | Time in the air in minutes. Typically null/blank if cancelled. |
| `FLIGHTS` | float/int | Flight count field (commonly `1` per record). Useful for aggregation. |
| `DISTANCE` | float | Route distance in miles. |

## Example Records

```csv
YEAR,MONTH,FL_DATE,OP_UNIQUE_CARRIER,TAIL_NUM,OP_CARRIER_FL_NUM,ORIGIN_AIRPORT_ID,ORIGIN,ORIGIN_CITY_NAME,ORIGIN_STATE_NM,DEST,DEST_CITY_NAME,DEST_STATE_NM,DEP_TIME,DEP_DELAY,ARR_TIME,ARR_DELAY,CANCELLED,AIR_TIME,FLIGHTS,DISTANCE
2025,4,2025-04-01,AA,N101NN,12,12892,LAX,"Los Angeles, CA",California,BOS,"Boston, MA",Massachusetts,828.0,-7.0,1617.0,-50.0,0.0,269.0,1.0,2611.0
2025,4,2025-04-01,AA,N101NN,1578,10721,BOS,"Boston, MA",Massachusetts,LAX,"Los Angeles, CA",California,1753.0,-7.0,2135.0,-8.0,0.0,380.0,1.0,2611.0
