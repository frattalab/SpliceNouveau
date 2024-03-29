import argparse
import pandas as pd
import glob
import os
import time
import random

#sbatch -t 24:00:00 --wrap="python3 generate_commands.py --output_dir auto_results/ --commands_file commands/uniprotkb_proteome_UP000000625_AND_revi_2024_01_18_commands.csv.gz --min_score 0.5"


def read_csv_folder(folder_path, min_score):
    """Reads all CSV files in a folder, excluding those with 'predictions' in the filename,
    and combines them into a single DataFrame.

    Args:
        folder_path (str): The path to the folder containing the CSV files.

    Returns:
        pandas.DataFrame: A DataFrame containing the combined data from the filtered CSV files.
    """

    csv_files = glob.glob(os.path.join(folder_path, '*'))

    filtered_files = []
    for file in csv_files:
        if 'predictions' not in file:  # Check if 'predictions' is in the filename
            filtered_files.append(file)

    if not filtered_files:
        print('no files')
        return []

    already_good_result = []

    for file in filtered_files:
        protein_name = file.split('/')[-1].split('Position')[0]

        df = pd.read_csv(file)
        if df['score'].any() > min_score:
            already_good_result.append(protein_name)
    print('\n\n\n\nhello')
    print(already_good_result)
    return already_good_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--commands_file', required=True)
    parser.add_argument('--min_score', default=1.75, type=float)
    args = parser.parse_args()

    commands_df = pd.read_csv(args.commands_file)

    while True:
        already_good = read_csv_folder(args.output_dir, args.min_score)

        filtered_df = commands_df[~commands_df['protein_name'].isin(already_good)]
        print(len(commands_df))
        print(len(filtered_df))

        if len(filtered_df) == 0:
            print('All complete')
            break

        # Pick a random command to run
        just_one_df = filtered_df.sample(n=1)

        random_command = just_one_df['command'].iloc[0]
        random_protein_name = just_one_df['protein_name'].iloc[0]
        print(random_command)

        # add a unique identifier
        unique_id = random.randint(1000000000, 9999999999)
        random_command_with_id = random_command[:-2] + str(unique_id) + '"'
        os.system(random_command_with_id)

        x = 0
        while x == 0:
            time.sleep(5)

            all_outputs = ' '.join(glob.glob(os.path.join(args.output_dir, '*.csv')))

            if str(unique_id) in all_outputs:
                x = 1
                time.sleep(5)
                print('completed ' + random_protein_name)
            else:
                print('still waiting')








if __name__ == '__main__':
    main()
