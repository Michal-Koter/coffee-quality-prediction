import pandas as pd
import yaml
from matplotlib import pyplot as plt


def main():
    df = pd.read_csv('data/coffee_data_cleaned.csv', index_col=0)

    df = df[['Country_of_Origin', 'In-Country_Partner', 'Variety', 'Processing_Method', 'Color', 'Total_Cup_Points']]

    countries = df['Country_of_Origin'].unique().tolist()
    countries_dict = {}
    for idx, country in enumerate(countries):
        countries_dict[country] = idx
    df['Country_of_Origin'] = df['Country_of_Origin'].map(countries_dict)

    partners = df['In-Country_Partner'].unique().tolist()
    partners_dict = {}
    for idx, partner in enumerate(partners):
        partners_dict[partner] = idx
    df['In-Country_Partner'] = df['In-Country_Partner'].map(partners_dict)

    varieties = df['Variety'].unique().tolist()
    varieties_dict = {}
    for idx, variety in enumerate(varieties):
        varieties_dict[variety] = idx
    df['Variety'] = df['Variety'].map(varieties_dict)

    methods = df['Processing_Method'].unique().tolist()
    methods_dict = {}
    for idx, method in enumerate(methods):
        methods_dict[method] = idx
    df['Processing_Method'] = df['Processing_Method'].map(methods_dict)

    colors = df['Color'].unique().tolist()
    colors_dict = {}
    for idx, color in enumerate(colors):
        colors_dict[color] = idx
    df['Color'] = df['Color'].map(colors_dict)

    label_to_id = {
        'Country_of_Origin': countries_dict,
        'In-Country_Partner': partners_dict,
        'Variety': varieties_dict,
        'Processing_Method': methods_dict,
        'Color': colors_dict
    }
    with open('label_to_id.yaml', "w") as file:
        yaml.dump(label_to_id, file)

    print("Data correlation:")
    print(df.corr())

    for col in df.columns:
        df[col].value_counts().sort_index().plot.bar()
        plt.title(f'Histogram {col}')
        plt.show()

    train_df = df.drop(index=df.index.values[::6])
    train_df = train_df.reset_index(drop=True)

    test_df = df.drop(index=train_df.index.values)
    test_df = test_df.reset_index(drop=True)

    train_df.to_csv('data/train_coffee_data.csv', index=False)
    test_df.to_csv('data/test_coffee_data.csv', index=False)

    print("Data prepared successfully")


if __name__ == '__main__':
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 230)
    main()
