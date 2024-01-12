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
    df['Country_of_Origin'] = df['Country_of_Origin'].fillna('Other')
    df['Farm_Name'] = df['Farm_Name'].fillna('Other')
    df['Mill'] = df['Mill'].fillna('Other')
    df['Company'] = df['Company'].fillna('Other')
    df['Region'] = df['Region'].fillna('Other')
    df['Producer'] = df['Producer'].fillna('Other')
    df['Region'] = df['Region'].fillna('Other')
    df['Owner'] = df['Owner'].fillna('Other')
    df['Variety'] = df['Variety'].fillna('Other')
    df['Processing_Method'] = df['Processing_Method'].fillna('Other')
    df['Color'] = df['Color'].fillna('Other')

    # Replace 'Country of Origin' with less than 10 occurrences
    countries_counts = df['Country_of_Origin'].value_counts()
    countries_to_replace = countries_counts[countries_counts < 10].index
    df.loc[df['Country_of_Origin'].isin(countries_to_replace), 'Country_of_Origin'] = 'Other'

    df['Farm_Name'] = df['Farm_Name'].fillna('Other')

    # Convert 'Bag Weight' to kg
    df[['Value', 'Unit']] = df['Bag_Weight'].str.extract(r'(\d+\.?\d*)\s*([a-zA-Z]*,?[a-zA-Z]*)')
    df['Value'] = pd.to_numeric(df['Value'])
    unit_conversion = {'lbs': 0.453592, 'kg': 1.0, 'kg,lbs': 1.0, '': 1.0}
    df['Bag_Weight'] = df['Value'] * df['Unit'].map(unit_conversion)
    df.drop(['Value', 'Unit'], axis=1, inplace=True)

    df['Variety'] = df['Variety'].mask(df['Variety'].str.contains('SL28' or 'Sl28'), 'SL28')
    df['Variety'] = df['Variety'].mask(df['Variety'].str.contains('SL34' or 'Sl34'), 'SL34')
    df['Variety'] = df['Variety'].mask(df['Variety'].str.contains('unknown'), 'Other')

    # Replace 'Variety' with less than 5 occurrences
    variety_counts = df['Variety'].value_counts()
    variety_to_replace = pd.DataFrame(variety_counts[variety_counts < 5]).index
    df.loc[df['Variety'].isin(variety_to_replace), 'Variety'] = 'Other'

    processing_methods_counts = df['Processing_Method'].value_counts()
    processing_methods_to_replace = pd.DataFrame(processing_methods_counts[processing_methods_counts < 5]).index
    df.loc[df['Processing_Method'].isin(processing_methods_to_replace), 'Processing_Method'] = 'Other'

    df['Moisture_Percentage'] = pd.to_numeric(df['Moisture_Percentage'], errors='coerce')
    median_value = df['Moisture_Percentage'].median()
    df['Moisture_Percentage'] = df['Moisture_Percentage'].replace(np.nan, median_value)
    q1 = df['Moisture_Percentage'].quantile(0.25)
    q3 = df['Moisture_Percentage'].quantile(0.75)
    qr = q3 - q1
    df.loc[df["Moisture_Percentage"] < q1 - 1.5 * qr, "Moisture_Percentage"] = median_value
    df.loc[df["Moisture_Percentage"] > q3 + 1.5 * qr, "Moisture_Percentage"] = median_value

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

    # print(df.isna().sum())
    # df.to_csv('data/coffee_data_cleaned.csv')


if __name__ == '__main__':
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 230)
    main()
