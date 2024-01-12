import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


def main():
    df = pd.read_csv('data/coffee_data_raw.csv', index_col=0)

    # delete column:
    # 'Lot Number' due to large number of Null
    # 'ICO Numer' is individual value
    df = df.drop(['Lot_Number', 'ICO_Number', 'Harvest_Year', 'Grading_Date', 'Altitude', 'Certification_Body',
                  'Certification_Address', 'Certification_Contact', 'Expiration'], axis=1)

    df = df.fillna(np.nan)

    df = clean_country(df)
    df = clean_farm(df)
    df = clean_mill(df)
    df = clean_company(df)
    df = clean_region(df)
    df = clean_producer(df)
    df = clean_num_bags(df)
    df = clean_bags_weight(df)
    df = clean_partner(df)
    df = clean_owner(df)
    df = clean_variety(df)
    df = clean_processing(df)
    df = clean_moisture(df)

    # df = clean_one_defect(df)
    # df = clean_two_defect(df)

    df = clean_quakers(df)
    df = clean_color(df)

    df = clean_total_points(df)

    df.to_csv('data/coffee_data_cleaned.csv')


def clean_country(df):
    df['Country_of_Origin'] = df['Country_of_Origin'].fillna('Other')

    # Replace 'Country of Origin' with less than 10 occurrences
    countries_counts = df['Country_of_Origin'].value_counts()
    countries_to_replace = countries_counts[countries_counts < 10].index
    df.loc[df['Country_of_Origin'].isin(countries_to_replace), 'Country_of_Origin'] = 'Other'

    return df


def clean_farm(df):
    df['Farm_Name'] = df['Farm_Name'].fillna('Other')

    return df


def clean_mill(df):
    df['Mill'] = df['Mill'].fillna('Other')
    return df


def clean_company(df):
    df['Company'] = df['Company'].fillna('Other')
    return df


def clean_region(df):
    df['Region'] = df['Region'].fillna('Other')
    return df


def clean_producer(df):
    df['Producer'] = df['Producer'].fillna('Other')
    return df


def clean_num_bags(df):
    return df


def clean_bags_weight(df):
    # Convert 'Bag Weight' to kg
    df[['Value', 'Unit']] = df['Bag_Weight'].str.extract(r'(\d+\.?\d*)\s*([a-zA-Z]*,?[a-zA-Z]*)')
    df['Value'] = pd.to_numeric(df['Value'])
    unit_conversion = {'lbs': 0.453592, 'kg': 1.0, 'kg,lbs': 1.0, '': 1.0}
    df['Bag_Weight'] = df['Value'] * df['Unit'].map(unit_conversion)
    df.drop(['Value', 'Unit'], axis=1, inplace=True)

    return df


def clean_partner(df):
    return df


def clean_owner(df):
    df['Owner'] = df['Owner'].fillna('Other')
    return df


def clean_variety(df):
    df['Variety'] = df['Variety'].fillna('Other')

    df['Variety'] = df['Variety'].mask(df['Variety'].str.contains('SL28' or 'Sl28'), 'SL28')
    df['Variety'] = df['Variety'].mask(df['Variety'].str.contains('SL34' or 'Sl34'), 'SL34')
    df['Variety'] = df['Variety'].mask(df['Variety'].str.contains('unknown'), 'Other')

    # Replace 'Variety' with less than 5 occurrences
    variety_counts = df['Variety'].value_counts()
    variety_to_replace = pd.DataFrame(variety_counts[variety_counts < 5]).index
    df.loc[df['Variety'].isin(variety_to_replace), 'Variety'] = 'Other'

    return df


def clean_processing(df):
    df['Processing_Method'] = df['Processing_Method'].fillna('Other')

    processing_methods_counts = df['Processing_Method'].value_counts()
    processing_methods_to_replace = pd.DataFrame(processing_methods_counts[processing_methods_counts < 5]).index
    df.loc[df['Processing_Method'].isin(processing_methods_to_replace), 'Processing_Method'] = 'Other'

    return df


def clean_moisture(df):
    df['Moisture_Percentage'] = pd.to_numeric(df['Moisture_Percentage'], errors='coerce')

    median_value = df['Moisture_Percentage'].median()

    df['Moisture_Percentage'] = df['Moisture_Percentage'].replace(np.nan, median_value)

    q1 = df['Moisture_Percentage'].quantile(0.25)
    q3 = df['Moisture_Percentage'].quantile(0.75)
    qr = q3 - q1
    df.loc[df["Moisture_Percentage"] < q1 - 1.5 * qr, "Moisture_Percentage"] = median_value
    df.loc[df["Moisture_Percentage"] > q3 + 1.5 * qr, "Moisture_Percentage"] = median_value

    return df


# def clean_one_defect(df):
#     return df
#
#
# def clean_two_defect(df):
#     return df


def clean_quakers(df):
    median_value = df['Quakers'].median()
    df['Quakers'] = df['Quakers'].replace(np.nan, median_value)

    # I wanted to use the IQR Method, but it caused all values to be set to 0
    #
    # q1 = df['Quakers'].quantile(0.25)
    # q3 = df['Quakers'].quantile(0.75)
    # qr = q3 - q1
    # df.loc[df["Quakers"] < q1 - 1.5 * qr, "Quakers"] = median_value
    # df.loc[df["Quakers"] > q3 + 1.5 * qr, "Quakers"] = median_value

    return df


def clean_color(df):
    df['Color'] = df['Color'].fillna('Other')

    df['Color'] = df['Color'].apply(lambda x: x.lower())
    df['Color'] = df['Color'].mask(df['Color'].str.contains('bluish-green'), 'blue-green')
    df['Color'] = df['Color'].mask(df['Color'].str.contains('greenish'), 'green')
    df['Color'] = df['Color'].mask(df['Color'].str.contains('yellow green'), 'yellow-green')
    df['Color'] = df['Color'].mask(df['Color'].str.contains('yellow- green'), 'yellow-green')
    df['Color'] = df['Color'].mask(df['Color'].str.contains('yello-green'), 'yellow-green')
    df['Color'] = df['Color'].mask(df['Color'].str.contains('yellowish'), 'yellow')
    df['Color'] = df['Color'].mask(df['Color'].str.contains('pale yellow'), 'yellow')
    df['Color'] = df['Color'].mask(df['Color'].str.contains('brownish'), 'brown')
    df['Color'] = df['Color'].mask(df['Color'].str.contains('browish-green'), 'brown')

    return df


def clean_total_points(df):
    index_to_remove = df[df['Total_Cup_Points'] == 0].index
    df = df.drop(index_to_remove, axis=0)

    return df

if __name__ == '__main__':
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 230)
    main()
