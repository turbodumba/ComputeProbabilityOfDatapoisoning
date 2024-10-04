import matplotlib.pyplot as plt

if __name__ == '__main__':
    file_path = 'Plots\\Combined_Scores.txt'
    with open(file_path, 'r') as file:
        data = file.read()

    lines = data.split('\n')
    original_scores = {}
    poisoned_scores = {}
    current_percentage = None

    for line in lines:
        if 'got this score' in line:
            parts = line.split('  got this score:  ')
            original_scores[parts[0]] = float(parts[1])
        elif 'Poisoned Datasets with manipulating percentage' in line:
            current_percentage = float(line.split(':  ')[1])
        elif 'got this difference' in line:
            parts = line.split('  got this difference:  ')
            dataset = parts[0].strip()
            score = float(parts[1]) + original_scores[dataset]
            if dataset not in poisoned_scores:
                poisoned_scores[dataset] = []
            poisoned_scores[dataset].append((current_percentage, score))

    for dataset, scores in poisoned_scores.items():
        percentages, differences = zip(*scores)
        plt.figure()
        plt.plot(percentages, differences, marker='o', label=f'{dataset} Poisoned Score')
        plt.axhline(y=original_scores[dataset], color='r', linestyle='--', label='Original Score')
        plt.xlabel('Manipulating Percentage')
        plt.ylabel('Poisoned Score')
        plt.title(f'Poisoned & Orginal score for {dataset}')
        plt.legend()
        plt.savefig(f'Plots\\{dataset}_differences_plot.jpg')
        plt.show()
