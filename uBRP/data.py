import torch
import os
import re
def transform_format(instance_file, H_plus):
    # Read the instance file
    with open(instance_file, 'r') as file:
        lines = file.readlines()

    # Extracting number_of_stacks and number_of_blocks
    num_stacks, num_blocks = map(int, lines[0].split())
    # Initializing the result list
    result = []

    # Loop through each row and transform the data
    for i in range(1, num_stacks + 1):
        block_values = list(map(lambda x: 1/int(x), lines[i].split()[1:]))
        row = block_values + [0] * H_plus
        result.append(row)

    return torch.tensor(result)
def process_files_with_regex(directory_path, file_regex, H_plus):
    # Use re to find files matching the specified regex pattern
    files = [file for file in os.listdir(directory_path) if re.search(file_regex, file)]
    transform_datas = []
    # Process each matching file
    for file_name in files:
        file_path = os.path.join(directory_path, file_name)
        transformed_data = transform_format(file_path,H_plus)
        transform_datas.append(transformed_data)
    return transform_datas
def data_from_caserta(file_regex="data3-3.*", H_plus=2):
    directory_path  = 'uBRP\\brp-instances-caserta-etal-2012\\CRPTestcases_Caserta'
    transform_datas = process_files_with_regex(directory_path, file_regex, H_plus)
    return transform_datas
if __name__ == '__main__':
    print(data_from_caserta())