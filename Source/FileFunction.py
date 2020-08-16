import numpy
import re

#This function takes the input text (string) and then line by line, cleans up the data by removing the special
#characters shown below. It will a return a string that is completely stripped of special characters and unwanted
#characters,as well as all lowercase characters
def data_cleanup(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

#This function pulls data from a positive data file and negative data file. It will pull in the data (x) and then
#clean the data by sending it to the function data_cleanup(sent) sentence by sentence. This function will return x_text
#which is the clean data set, and it will also return y, which is a collection of the positive and negative labels
#in the dataset. Y will be returned as a numpy array
def import_data(positive_data_file, negative_data_file):
    # Load data from files pos and neg datasets
    #Open pos dataset and read it line by line
    positive_examples = list(open(positive_data_file, "r").readlines())
    #Strip the data contained in the pos dataset
    positive_examples = [s.strip() for s in positive_examples]
    #Open neg dataset adn read it line by line
    negative_examples = list(open(negative_data_file, "r").readlines())
    #Strip the dadta contained in the pos dataset
    negative_examples = [s.strip() for s in negative_examples]
    #Split data contained in x_text by words (tokenization)
    #Compbine the positive and negative samples and store result in x_text
    x_text = positive_examples + negative_examples
    #Clean the data in x_text by sending it to data_cleanup() sentence by sentence in the data of x_text
    x_text = [data_cleanup(sent) for sent in x_text]
    # Generate labels
    #Create positive labels from the positive dataset
    positive_labels = [[0, 1] for _ in positive_examples]
    #Create negative labels from the negative dataset
    negative_labels = [[1, 0] for _ in negative_examples]
    #Concat results in a numpy array stored in y. This is a collection of both positive and negative labels
    y = numpy.concatenate([positive_labels, negative_labels], 0)
    #Returns both x_text and y
    return [x_text, y]

#This function returns batches for the training procedure to traverse through. It will date the data collection of
#both x and y, the size of each batch, the number of epochs, and shuffle which by default True.
def iteration_bat(data, sz_bat, num_epochs, shuffle=True):
    #Convert the received data into a numpy array
    data = numpy.array(data)
    #Determine the size of the received data
    data_size = len(data)
    #Determine the number of batches per epoch from the length of data divided by the size of each batch, then add 1
    epoch_bat = int((len(data)-1)/sz_bat) + 1
    #Travers through each epoch a total of num_epochs times
    for epoch in range(num_epochs):
        #This if/else statement shuffles the data if the value of shuffle is True. By default, it is true
        if shuffle:
            #Determine the index for the shuffled data
            index = numpy.random.permutation(numpy.arange(data_size))
            dat_shuf = data[index]
        #If value of shuffle is false, the data will not be shuffled
        else:
            dat_shuf = data
        #for each batch in the range number of epochs per batch, determine the start and end index for the shuffled
        #data
        for z in range(epoch_bat):
            start = z * sz_bat
            end = min((z + 1) * sz_bat, data_size)
            yield dat_shuf[start:end]
