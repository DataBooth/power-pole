library(weatherOz)

# Get daily summaries (closest available)
obs_data <- get_dpird_summaries(
  station_code = "Observatory Hill",  # Replace with actual station code
  start_date = "2018-01-01",
  end_date = "2023-12-31",
  interval = "daily",
  values = c("air_temp")
)

# Example output (daily temps)
head(obs_data)
