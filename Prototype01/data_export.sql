CREATE TEMPORARY VIEW data_export AS (
    SELECT
        EXTRACT(EPOCH FROM fmc.fmc_date)  * 1000 as fmc_date,

        fmc.total_odometer,
        fmc.gsm_signal,
        fmc.speed,
        fmc.number_of_dtc,
        fmc.engine_load,
        fmc.coolant_temperature,
        fmc.intake_map,
        fmc.engine_rpm,
        fmc.vehicle_speed,
        fmc.intake_air_temperature,
        fmc.maf,
        fmc.throttle_position,
        fmc.runtime_since_engine_start,
        fmc.distance_traveled_mil_on,
        fmc.direct_fuel_rail_pressure,
        fmc.commanded_egr,
        fmc.egr_error,
        fmc.fuel_level,
        fmc.distance_since_codes_clear,
        fmc.barometic_pressure,
        fmc.control_module_voltage,
        fmc.absolute_load_value,
        fmc.ambient_air_temperature,
        fmc.time_since_codes_cleared,
        fmc.absolute_fuel_rail_pressure,
        fmc.engine_oil_temperature,
        fmc.fuel_injection_timing,
        fmc.fuel_rate,
        fmc.external_voltage,
        fmc.battery_voltage,
        fmc.battery_current,
        fmc.gnss_status,
        fmc.data_mode,
        fmc.gnss_pdop,
        fmc.gnss_hdop,
        fmc.sleep_mode,
        fmc.ignition,
        fmc.movement,
        fmc.active_gsm_operator,
        fmc.green_driving_type,
        fmc.unplug,
        fmc.green_driving_value,
        fmc.sped
    FROM
        public.fm_telemetry_volvo fmc
        /* public.fm_telemetry_1323 fmc */
        join fm_motion_unit fmmu on
            fmmu.fmobjsrv_id = 5439
            /* fmmu.fmobjsrv_id = 5525 */
            and fmmu.fmmu_type = 1
            and fmc.fmc_date between fmmu.fmc_begin_date and fmmu.fmc_end_date
    ORDER BY
        fmc.fmc_date
);

\copy (select * from data_export) to './telemetry_volvo.csv' with csv header;