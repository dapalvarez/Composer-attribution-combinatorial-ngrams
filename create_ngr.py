
import pathlib2 as pl2
from os import listdir
from os.path import join, splitext


def load_file(file_path, rows, separator):
    arr_info = []
 
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            line_info = line.split(separator)
            if len(line_info) != rows:
                print('error in line', line,  'number of rows different to', rows)
                exit(1)
            arr_info.append(line_info)
    
    return arr_info


def get_regular_ngrams(matrix_info, ngram_order, is_class):
    
    all_ngrams = ''
    
    for j in range(len(matrix_info[0])):

        ngram_list = []

        for i in range(0, len(matrix_info), 1):

            elem = matrix_info[i][j]
            if elem != 'r' and is_class:
                elem = int(elem) % 12
            ngram_list.append(str(elem))

            if len(ngram_list) == ngram_order:
                all_ngrams = all_ngrams+' '.join(ngram_list)+'|'
                del ngram_list[0]

    return all_ngrams[0: len(all_ngrams)-1]


def get_combinatorial_ngrams(matrix_info, ngram_order, is_class):
    
    all_ngrams = ''
    
    for i in range(0, len(matrix_info) - ngram_order + 1, 1):
        matrix_slice = matrix_info[i: i+ngram_order]
        ngrams_slice = get_combinations_from_matrix(matrix_slice, is_class)
        new_ngrams = '|'.join(ngrams_slice)
        all_ngrams = all_ngrams+new_ngrams+'|'
    return all_ngrams[0: len(all_ngrams)-1]


def get_combinations_from_matrix(matrix, is_class):
        
    arr_ngrams = [""]
    rows = len(matrix)
    columns = len(matrix[0])
    
    for i in range(rows):
        new_arr = []
        for j in range(columns):
            size_arr_ngrams = len(arr_ngrams)
            
            for k in range(size_arr_ngrams):
                separator = ' '
                if i == rows - 1:
                    separator = ''
                elem = matrix[i][j]
                if elem != 'r' and is_class:
                    elem = int(elem)%12
                new_arr.append(arr_ngrams[k]+str(elem)+separator)
        arr_ngrams=new_arr
    
    return arr_ngrams


# ngram_type: {'combinatorial', 'regular'} ; pitch_type: {'class', 'midi'} ; ngram_order: {1, 2, 3, 4}
def convert_pitch_data_in_ngrams(ngram_type, pitch_type, ngram_order):
    
    straight=True
    
    path_dataset = 'long_rep'
    separator = ' '
    rows = 4
    
    ngr_base_path = join('ngrams', ngram_type, pitch_type, str(ngram_order))
    print('Processing directory:', ngr_base_path)
    
    for candidate in listdir(path_dataset):
        print('\nCandidate:', candidate)
        
        path_candidate = join(path_dataset, candidate)
        for score in listdir(path_candidate):
            
            if splitext(score)[1][1:] != 'ptch':
                continue 
            print('Score:', score)
            
            path_score = join(path_candidate, score)
            matrix_pitchs = load_file(path_score, rows, separator)
            
            if ngram_type == 'regular':
                ngrams = get_regular_ngrams(matrix_pitchs, ngram_order, pitch_type == 'class') 
            else:
                ngrams = get_combinatorial_ngrams(matrix_pitchs, ngram_order, pitch_type == 'class')
            print('Ngrams string size:', len(ngrams))
            
            output_file = join(ngr_base_path, candidate, splitext(score)[0]+'.ngr')
            pl2.Path(join(ngr_base_path, candidate)).mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as ff:
                ff.write(ngrams)


def run_all():
    
    convert_pitch_data_in_ngrams('regular', 'class', 1)
    convert_pitch_data_in_ngrams('regular', 'class', 2)
    convert_pitch_data_in_ngrams('regular', 'class', 3)
    convert_pitch_data_in_ngrams('regular', 'class', 4)
    
    convert_pitch_data_in_ngrams('regular', 'midi', 1)
    convert_pitch_data_in_ngrams('regular', 'midi', 2)
    convert_pitch_data_in_ngrams('regular', 'midi', 3)
    convert_pitch_data_in_ngrams('regular', 'midi', 4)
    
    convert_pitch_data_in_ngrams('combinatorial', 'class', 1)
    convert_pitch_data_in_ngrams('combinatorial', 'class', 2)
    convert_pitch_data_in_ngrams('combinatorial', 'class', 3)
    convert_pitch_data_in_ngrams('combinatorial', 'class', 4)
    
    convert_pitch_data_in_ngrams('combinatorial', 'midi', 1)
    convert_pitch_data_in_ngrams('combinatorial', 'midi', 2)
    convert_pitch_data_in_ngrams('combinatorial', 'midi', 3)
    convert_pitch_data_in_ngrams('combinatorial', 'midi', 4)
    
    print('\n:)')

run_all()