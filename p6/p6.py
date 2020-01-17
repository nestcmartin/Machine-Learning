import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import warnings
from sklearn import svm
warnings.filterwarnings('ignore')

import re
import nltk
import nltk.stem.porter
import codecs

# Plot functions ----------------

def plot_data(X, y, title):
    # +: elementos positivos / o: elementos negativos
    pos = y == 1
    neg = y == 0

    plt.title(title)
    plt.plot(X[:,0][pos], X[:,1][pos], "k+") # Positivos, color = black, shape = +
    plt.plot(X[:,0][neg], X[:,1][neg], "yo") # Negativos, color = yellow, shape = o
    plt.show()

def visualizeBoundryLinear(X, y, model, title):
    # para visualizar la frontera de decision linear aprendida
    w = model.coef_[0]
    a = -w[0] / w[1]

    xx = np.array([X[:, 0].min(), X[:, 0].max()])
    yy = a * xx - (model.intercept_[0]) / w[1]

    plt.plot(xx, yy, 'b-') # linea separadora

    plot_data(X, y, title)

def visualizeBoundry(X, y, model, sigma, title):
    # para visualizar la frontera de decision aprendida
    x1plot = np.linspace(X[:,0].min(), X[:,0].max(), 100).T
    x2plot = np.linspace(X[:,1].min(), X[:,1].max(), 100).T
    X1, X2 = np.meshgrid(x1plot, x2plot)

    vals = np.zeros(X1.shape)
    for i in range(X1.shape[1]):
        this_X = np.column_stack((X1[:, i], X2[:, i]))
        vals[:, i] = model.predict(gaussianKernel(this_X, X, sigma))

    plt.contour(X1, X2, vals, colors="b", levels=[0,0])
    plot_data(X, y, title)

# ------------------------------

# Proceso de emails ------------

def preProcess(email):
    
    hdrstart = email.find("\n\n")
    if hdrstart != -1:
        email = email[hdrstart:]

    email = email.lower()
    # Strip html tags. replace with a space
    email = re.sub('<[^<>]+>', ' ', email)
    # Any numbers get replaced with the string 'number'
    email = re.sub('[0-9]+', 'number', email)
    # Anything starting with http or https:// replaced with 'httpaddr'
    email = re.sub('(http|https)://[^\s]*', 'httpaddr', email)
    # Strings with "@" in the middle are considered emails --> 'emailaddr'
    email = re.sub('[^\s]+@[^\s]+', 'emailaddr', email)
    # The '$' sign gets replaced with 'dollar'
    email = re.sub('[$]+', 'dollar', email)
    return email


def email2TokenList(raw_email):
    """
    Function that takes in a raw email, preprocesses it, tokenizes it,
    stems each word, and returns a list of tokens in the e-mail
    """

    stemmer = nltk.stem.porter.PorterStemmer()
    email = preProcess(raw_email)

    # Split the e-mail into individual words (tokens) 
    tokens = re.split('[ \@\$\/\#\.\-\:\&\*\+\=\[\]\?\!\(\)\{\}\,\'\"\>\_\<\;\%]',
                      email)

    # Loop over each token and use a stemmer to shorten it
    tokenlist = []
    for token in tokens:

        token = re.sub('[^a-zA-Z0-9]', '', token)
        stemmed = stemmer.stem(token)
        #Throw out empty tokens
        if not len(token):
            continue
        # Store a list of all unique stemmed words
        tokenlist.append(stemmed)

    return tokenlist

def getVocabDict(reverse=False):
    """
    Function to read in the supplied vocab list text file into a dictionary.
    Dictionary key is the stemmed word, value is the index in the text file
    If "reverse", the keys and values are switched.
    """
    vocab_dict = {}
    with open("vocab.txt") as f:
        for line in f:
            (val, key) = line.split()
            if not reverse:
                vocab_dict[key] = int(val)
            else:
                vocab_dict[int(val)] = key

    return vocab_dict

# ------------------------------

# Parte 1 ----------------------

def gaussianKernel(X1, X2, sigma):
    result = np.zeros((X1.shape[0], X2.shape[0]))

    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            x1 = x1.ravel()
            x2 = x2.ravel()
            result[i, j] = np.exp(-np.sum(np.square(x1 - x2)) / (2 * (sigma**2)))
    
    return result

def svmGaussianTrain(X, y, C, tol, max_passes, sigma=None):
    clf = svm.SVC(C=C, kernel='precomputed', tol=tol, max_iter=max_passes)
    return clf.fit(gaussianKernel(X, X, sigma=sigma), y)

def svmLinearTrain(X, y, C, tol, max_passes, sigma=None):
    clf = svm.SVC(C=C, kernel='linear', tol=tol, max_iter=max_passes)
    return clf.fit(X, y)

def first_dataset():
    # Primer dataset:
    data1 = loadmat("ex6data1.mat")
    X = data1['X']
    y = data1['y']
    Y_ravel = y.ravel()
    
    C = 1
    model = svmLinearTrain(X, Y_ravel, C, 1e-3, -1)
    visualizeBoundryLinear(X, Y_ravel, model, "(Data1 - Linear) Frontera con C = 1")
    C = 100
    model = svmLinearTrain(X, Y_ravel, C, 1e-3, -1)
    visualizeBoundryLinear(X, Y_ravel, model, "(Data1 - Linear) Frontera con C = 100")

def second_dataset():
    # Segundo dataset:
    data2 = loadmat("ex6data2.mat")
    X = data2['X']
    y = data2['y']
    Y_ravel = y.ravel()

    C = 1
    sigma = 0.1
    model = svmGaussianTrain(X, Y_ravel, C, 1e-3, 100, sigma)
    visualizeBoundry(X, Y_ravel, model, sigma, "(Data2 - Gaussian) Frontera aprendida")

def third_dataset():
    data3 = loadmat("ex6data3.mat")
    X = data3['X']
    y = data3['y']
    Xval = data3['Xval']
    yval = data3['yval']
    Y_ravel = y.ravel()

    predictions = dict()
    for C in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:
        for sigma in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:
            model = svmGaussianTrain(X, Y_ravel, C, 1e-5, -1, sigma)
            prediction = model.predict(gaussianKernel(Xval, X, sigma))
            predictions[(C, sigma)] = np.mean((prediction != yval).astype(int))

    C, sigma = min(predictions, key = predictions.get)

    model = svmGaussianTrain(X, Y_ravel, C, 1e-5, -1, sigma=sigma)
    visualizeBoundry(X, Y_ravel, model, sigma, "(Data3 - Gaussian) Frontera aprendida")

def part_one():
    print("Primer dataset (ex6data1.mat)...")
    first_dataset()

    print("Segundo dataset (ex6data2.mat). Puede tardar unos minutos...")
    second_dataset()

    print("Tercer dataset (ex6data3.mat). Puede tardar unos minutos...")
    third_dataset()

# ------------------------------

# Parte 2 ----------------------

def tokenList2wordIndices(token_list, vocab_dict):
    word_indices = []

    for token in token_list:
        if token in vocab_dict:
            idx = vocab_dict[token]
            word_indices.append(idx)
    
    return word_indices


def part_two():
    # Ejemplo de reducir un email a sus atributos para poder
    # clasificarlo como spam o ham mediante SVM

    email_contents = codecs.open('spam/0001.txt', 'r', encoding='utf-8', errors='ignore').read()
    email_contents = preProcess(email_contents)
    email_token_list = email2TokenList(email_contents)

    vocab_dict = getVocabDict() # 1899 elementos
    email_word_indices = tokenList2wordIndices(email_token_list, vocab_dict)

    email_attributes = np.zeros((1899,), dtype=int)

    for idx in email_word_indices:
        email_attributes[idx] = 1

    print("email 'spam/0001.txt' atributtes: ")
    print(email_attributes)

# ------------------------------    

part_two()