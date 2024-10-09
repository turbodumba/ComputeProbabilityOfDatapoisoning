import os

import numpy as np
import pandas as pd


# Helper method or the out of distribution poisoning method, which fills the categorical columns with random possible values.
def fill_categorical_with_random_sample(dataset, columns_to_fill):
    for column in columns_to_fill:
        non_null_values = dataset[column].dropna().values
        random_samples = np.random.choice(non_null_values, size=len(dataset))
        dataset[column] = dataset[column].fillna(pd.Series(random_samples))
    return dataset


# Method to poison the labels of the given dataset, with the given flip percentage.
def poison_labels(dataset, columns_to_poison, flip_percentage=0.1):
    # Iterating over the given columns.
    for label_column in columns_to_poison:
        # Producing a list of valid indices, which could be flipped.
        valid_indices = dataset[~dataset[label_column].isna()].index
        num_poisoned = int(flip_percentage * len(valid_indices))
        # Producing a list of indices which will be flipped with the help of np.random.choice.
        flip_indices = np.random.choice(valid_indices, size=num_poisoned, replace=False)
        possible_labels = dataset[label_column].dropna().unique().tolist()

        # Iterating over the indices which are to be flipped.
        for idx in flip_indices:
            current_label = dataset.loc[idx, label_column]
            # Choosing a random label out of the list of possible labels for this column, without the current label
            new_label = np.random.choice([label for label in possible_labels if label != current_label])
            # Actually setting the new label.
            dataset.loc[idx, label_column] = new_label

    return dataset


# Method to poison the numerical columns given with noise, given the percentage of rows to poison and the level at which the noise should be generated.
def poison_with_random_noise(dataset, columns_to_poison, noise_percentage=0.1, noise_level=0.05):
    # Deciding randomly which indices to actually poison using np.random.choice.
    num_poisoned = int(noise_percentage * len(dataset))
    noise_indices = np.random.choice(dataset.index, size=num_poisoned, replace=False)

    # Going over the columns one by one poisoning the chosen indices.
    for column in columns_to_poison:
        # Generating the noise using the noise level given as a parameter.
        noise = np.random.normal(0, noise_level, size=num_poisoned)
        # Checking whether the indices have a float or int value and adjusting the noise accordingly.
        if pd.api.types.is_integer_dtype(dataset[column]):
            noise = noise.astype(int)
        elif pd.api.types.is_float_dtype(dataset[column]):
            noise = noise.astype(float)
        # Adding the noise to the poisoned indices.
        dataset.loc[noise_indices, column] += noise

    return dataset


# Method which poisons the numerical columns with data that is outside the distribution of the other data and adding these as new rows/datapoints.
def poison_with_out_of_distribution_data(dataset, columns_to_poison, categoricalColumns, ood_percentage=0.1):
    # Calculating how many poisoned rows needs to be added
    num_poisoned = int(ood_percentage * len(dataset))
    ood_data = {}
    # Iterating through the dataset column by column
    for column in columns_to_poison:
        # Calculating the data point for the given column which will be added.
        ood_data_point = np.random.uniform(dataset[column].min() * 1.5, dataset[column].max() * 1.5, num_poisoned)
        # Checking whether the column has a float or int value in general and adjusting the data point accordingly.
        if pd.api.types.is_integer_dtype(dataset[column]):
            ood_data_point = ood_data_point.astype(int)
        elif pd.api.types.is_float_dtype(dataset[column]):
            ood_data_point = ood_data_point.astype(float)
        ood_data[column] = ood_data_point
    ood_df = pd.DataFrame(ood_data)
    # Adding the poisoned data point to the dataset which will be returned.
    poisoned_dataset = pd.concat([dataset, ood_df], ignore_index=True)

    # Filling in the missing categorical columns to complete the poisoning method.
    poisoned_dataset = fill_categorical_with_random_sample(poisoned_dataset, categoricalColumns)
    return poisoned_dataset


# Helper method to combine the two numerical poison methods into one.
def poison_numerical_columns(dataset, columns_to_poison, manipulation_percentage, noise_level, categoricalColumns):
    poisoned_dataset = poison_with_random_noise(dataset, columns_to_poison, manipulation_percentage,
                                                noise_level)
    poisoned_dataset = poison_with_out_of_distribution_data(poisoned_dataset, columns_to_poison, categoricalColumns,
                                                            manipulation_percentage)
    return poisoned_dataset


# Main method of this class, which coordinates the poisoning of the dataset.
def createPoisonedDatasets(filepath, targetColumns, categoricalColumns, numericalColumns, manipulation_percentage=0.1,
                           noise_level=0.01):
    # Setting random seed for testing purposes can help when checking to see if the results change on the scoring side.
    # np.random.seed(42)
    # Get the absolute path to the Datasets folder
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    datasets_dir = os.path.join(base_dir, 'Datasets')

    # Change directory to the dataset folder
    os.chdir(datasets_dir)

    # Read the CSV file and poison it with the given parameters
    dataset = pd.read_csv(filepath)
    dataset = poison_labels(dataset, targetColumns, manipulation_percentage)
    dataset = poison_numerical_columns(dataset, numericalColumns, manipulation_percentage, noise_level,
                                       categoricalColumns)
    # Save the poisoned file with the correct name.
    dataset.to_csv(filepath.replace('.csv', '_Poisoned.csv'), index=False)
    return None


# This is a helper method that could be used to delete the poisoned dataset at the end of the scoring algorithm, but isn't used.
def deletePoisonedData(file_paths):
    # Get the absolute path to the Datasets folder
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    datasets_dir = os.path.join(base_dir, 'Datasets')
    # Change directory to the dataset folder
    os.chdir(datasets_dir)
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)
