-- HPNow experiment database schema
-- Run this once in the Supabase SQL editor to create the tables.

CREATE TABLE IF NOT EXISTS runs (
    id              uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    source_key      text        UNIQUE NOT NULL,   -- dedup key: "formal::stack_id::date_start"
    tab_name        text        NOT NULL,           -- original sheet tab name (test station)
    station_id      text,                           -- display name for the station
    operator        text,
    stack_id        text,
    date_start      text,                           -- kept as text; format varies across sheets
    project         text,
    aim             text,
    cabinet         text,
    n_cells         integer,
    cell_area_cm2   numeric,
    current_ma_cm2  numeric,
    gdl             text,
    foam_grid       text,
    operation_note  text,
    is_informal     boolean     NOT NULL DEFAULT false,
    migrated_at     timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS measurements (
    id                          bigserial   PRIMARY KEY,
    run_id                      uuid        NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    row_order                   integer,            -- original row position within the run
    -- Time
    time_h                      numeric,            -- "Time (hours)"
    time_s                      numeric,            -- "Time (seconds)"
    date_col                    text,               -- "Date" / "Date "
    time_of_day                 text,               -- "time"
    -- Electrical
    efficiency_pct              numeric,            -- "Efficiency (%)"
    current_a                   numeric,            -- "Current (A)"
    voltage_v                   numeric,            -- "Voltage (V)"
    avg_voltage_v               numeric,            -- "Average V (V)"
    current_density_ma_cm2      numeric,            -- "Current density (mA/cm²)"
    hfr                         numeric,            -- "HFR"
    -- Chemical / peroxide
    h2o2_current_a              numeric,            -- "H2O2 current (A)"
    h2o2_current_density_ma_cm2 numeric,            -- "H2O2 current density (mA/cm²)"
    strip_1                     numeric,            -- "Strip 1"
    strip_2                     numeric,            -- "Strip 2"
    peroxide_in_di              numeric,            -- "Peroxide in DI water"
    -- Process
    gas_lpm                     numeric,            -- "Gas (LPM)"
    water_flow_ml_s             numeric,            -- "Water flow (mL/s)"
    conductivity_us_cm          numeric,            -- "Conductivity (µS/cm)"
    diff_pressure_mbar          numeric,            -- "Diff Pressure (mbar)"
    anode_flow_ml_s             numeric,            -- "Anode flow (mL/s)"
    -- Temperature
    stk_temp_an_out             numeric,            -- "STK temp An out"
    stk_temp_ca_out             numeric,            -- "STK temp Ca out"
    -- Throughput
    throughput_g_h              numeric,            -- "Throughput (g/h)"
    avg_throughput_g_h          numeric,            -- "Avg. throughput (g/h)"
    -- Overflow for any columns not in the fixed list above
    extra_data                  jsonb
);

CREATE INDEX IF NOT EXISTS idx_measurements_run_id ON measurements(run_id);
