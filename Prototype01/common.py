from pandas import read_csv


def load_data_set(name, limit=0):
    df = read_csv(name, index_col=0)

    df = df.drop(columns=[
        'total_odometer',
        'gsm_signal',
        'number_of_dtc',
        'vehicle_speed',
        'runtime_since_engine_start',
        'distance_traveled_mil_on',
        'fuel_level',
        'distance_since_codes_clear',
        'absolute_load_value',
        'time_since_codes_cleared',
        'absolute_fuel_rail_pressure',
        'engine_oil_temperature',
        'fuel_injection_timing',
        'fuel_rate',
        'battery_current',
        'gnss_status',
        'data_mode',
        'gnss_pdop',
        'gnss_hdop',
        'sleep_mode',
        'ignition',
        'movement',
        'active_gsm_operator',
        'green_driving_type',
        'unplug',
        'green_driving_value',
        'sped'])

    if limit > 0:
        df = df.iloc[:limit, :]

    df = df.interpolate(method='linear', axis=0)

    df = df.fillna(0)

    return df