import pandas as pd
import re


def prepare_df_dot():
    df_dot = pd.read_csv('data/kaggle/arabica_data_cleaned.csv', index_col=0)
    df_dot = df_dot.drop(['Species', 'Owner', 'altitude_low_meters', 'altitude_high_meters', 'altitude_mean_meters', 'unit_of_measurement'], axis=1)

    original_labels = df_dot.columns.values
    new_labels = [re.sub('\.', ' ', label) for label in original_labels]
    label_mapper = dict(zip(original_labels, new_labels))
    label_mapper['Cupper.Points'] = 'Overall'
    label_mapper['In.Country.Partner'] = 'In-Country Partner'
    label_mapper['Moisture'] = 'Moisture Percentage'
    label_mapper['Owner.1'] = 'Owner'
    df_dot = df_dot.rename(label_mapper, axis='columns')

    return df_dot


def prepare_df_space():
    df_space = pd.read_csv('data/kaggle/df_arabica_clean.csv', index_col=0)
    df_space = df_space.drop(['ID', 'Defects', 'Status'], axis=1)

    return df_space


def main():
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 230)

    df_dot = prepare_df_dot()
    df_space = prepare_df_space()

    labels_1 = df_dot.columns.tolist()
    labels_2 = df_space.columns.tolist()

    if len(labels_1) != len(labels_2):
        print('Different number of columns!')
        print("Labels 1:", len(labels_1))
        print(labels_1)
        print("Labels 2:", len(labels_2))
        print(labels_2)
        return None

    if labels_1 != labels_2:
        print('Different names of columns!')
        print(labels_1 == labels_2)
        print("Labels 1:")
        print(labels_1)
        print("Labels 2:")
        print(labels_2)
        return None

    df = df_dot._append(df_space, ignore_index=True)

    original_labels = df.columns.values
    new_labels = [re.sub(' ', '_', label) for label in original_labels]
    label_mapper = dict(zip(original_labels, new_labels))
    df = df.rename(label_mapper, axis='columns')

    df.to_csv('data/coffee_data_raw.csv')

    print("Data merged successfully!")


if __name__ == "__main__":
    main()