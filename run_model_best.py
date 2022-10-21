
from os import listdir
from os.path import join, splitext

from nltk.tokenize import RegexpTokenizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC


# ngram_type: {'combinatorial', 'regular'} ; pitch_type: {'class', 'midi'} ; ngram_order: {'1', '2', '3', '4'}
def run_models(ngram_type, pitch_type, ngram_order):

    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
        
    path_dataset = join('ngrams', ngram_type, pitch_type, ngram_order)
    print('Processing directory:', path_dataset)

    train_scores = []
    train_labels = []
    score_names = []
    
    # read data
    for candidate in listdir(path_dataset):
        
        path_candidate = join(path_dataset, candidate)
        for ngr_score in listdir(path_candidate):
            
            if splitext(ngr_score)[1][1:] != 'ngr':
                continue 
            
            path_score = join(path_candidate, ngr_score)
            with open(path_score, 'r') as f:
                train_scores.append(f.read())
                train_labels.append(candidate)
                score_names.append(ngr_score)
    
    # get results
    result, nfeats = my_loo(train_scores, train_labels, score_names)
    print(result, '\t', nfeats)

def my_loo(train_scores_, train_labels_, score_names_):

    sum_feats = 0
    counter = 0
    for i in range(len(train_scores_)):

        test_data = [train_scores_[i]]   
        test_label = [train_labels_[i]]

        train_data = []
        train_labels = []

        for j in range(len(train_scores_)):
            if j != i:
                train_data.append(train_scores_[j])
                train_labels.append(train_labels_[j])

        tokenizer_ = RegexpTokenizer(r'[|]+', gaps=True)
        vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), lowercase=False, binary=False, \
                         tokenizer = tokenizer_.tokenize, max_features = 5000, min_df = 0.4, max_df = 0.8)
        transformer = TfidfTransformer(norm='l2', use_idf=False, smooth_idf=False, sublinear_tf=True)


        train_data = vectorizer.fit_transform(train_data)
        test_data = vectorizer.transform(test_data)

        train_data = transformer.fit_transform(train_data)
        test_data = transformer.transform(test_data)

        train_data = train_data.toarray()
        test_data = test_data.toarray()

        number_feats = len(vectorizer.get_feature_names())
        sum_feats += number_feats
                
        model = SVC(kernel='linear', C=10000)
        model.fit(train_data, train_labels)
        predictions=model.predict(test_data)

        if predictions == test_label:
            counter += 1
        else:
            print('Wrong', i, score_names_[i])
    
    # Accuracy, Mean number of features
    return round((counter*100/len(train_scores_)), 2), round((sum_feats/len(train_scores_)), 0)


def run_all():
    run_models('combinatorial', 'midi', '3')
    
    print('\n:)')

run_all()