import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Set the file which should be used to create the plots
    file_path = 'Plots\\Combined_Scores.txt'
    with open(file_path, 'r') as file:
        data = file.read()

    lines = data.split('\n')
    original_scores = {}
    poisoned_scores = {}
    current_percentage = None

    # Going over the read in file line by line and extracting the scores and percentages according to the text in the file.
    for line in lines:
        if 'got this score' in line:
            parts = line.split(' got this score: ')
            original_scores[parts[0]] = float(parts[1])
        elif 'Poisoned Datasets with manipulating percentage' in line:
            current_percentage = float(line.split(':  ')[1])
        elif 'got this poisoned score' in line:
            parts = line.split(' got this poisoned score: ')
            dataset = parts[0].strip()
            score = float(parts[1])
            if dataset not in poisoned_scores:
                poisoned_scores[dataset] = []
            poisoned_scores[dataset].append((current_percentage, score))

    # Creating a plot for each of the datasets read in by the for loop before. Showing both the original score, the different poisoned scores and the manipulation percentages that correspond to them.
    for dataset, scores in poisoned_scores.items():
        percentages, differences = zip(*scores)
        # Creating the plot and adding the important information.
        plt.figure()
        plt.plot(percentages, differences, marker='o', label=f'{dataset} Poisoned Score')
        plt.axhline(y=original_scores[dataset], color='r', linestyle='--', label='Original Score')
        # Setting the different labels, title and generating the legend.
        plt.xlabel('Manipulating Percentage')
        plt.ylabel('Poisoned Score')
        plt.title(f'Poisoned & Orginal score for {dataset}')
        plt.legend()
        # Saving the created plot
        plt.savefig(f'Plots\\{dataset}_differences_plot.jpg')
        # Displaying the created plot
        plt.show()
