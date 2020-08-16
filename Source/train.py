import tensorflow
import numpy
import os
import time
import datetime
import FileFunction
import TextConvNet
from tensorflow.contrib import learn

#%%Parameters%%#

# Data loading params
#Percent of data being split for validation set
dev_percent = 0.1
#Data file for positive examples
pos_dat_file = './data/rt-polaritydata/rt-polarity.pos'
#Data file for negative examples
neg_dat_file = './data/rt-polaritydata/rt-polarity.neg'

#-->Hyperparameters
#Embedding size
embed_size = 128
#String for filter size of 3, 4, and 5
temp = "3,4,5"
#Filter size
fil_sz = list(map(int, temp.split(",")))
#Number of filters
fil_num = 128
#Original dropout keep probability
drop_keep = 0.5
#Initialize l2_lambda_reg as 0.0
l2 = 0.0
#Learning Rate
learning_rate = 1e-5

#-->Training parameters
#Size of each batch
batch_sz = 64
#Total number of epochs
epoch_num = 200
#Validation evaluations
eval_on_dev = 100
#Checkpoint evaluations
ckpt = 100
#Total number of checkpoints
ckpt_num = 5

#-->Misc Parameters
soft_pl = True
log_pl = False

#%%Prep Data for Text Classification%%#

#-->Load data
print("\nLoad data from input files...")
x_text, y = FileFunction.import_data(pos_dat_file, neg_dat_file)

#-->Build vocabulary
print("\nBuilding Vocabulary...")
#Determine the maximum length of the document from the combined data of positive and negative samples
max_document_length = max([len(x.split(" ")) for x in x_text])
#Process the vocabulary using VocabularyProcessory from tensorflow.contrib learn
process_voc = learn.preprocessing.VocabularyProcessor(max_document_length)
#Convert data into numpy array and use fit_transform from the vocabulary above
x = numpy.array(list(process_voc.fit_transform(x_text)))

#-->Randomly shuffle data
#Random number generator through numpy
numpy.random.seed(10)
#Create index for start and end for randomly shuffled data
shuffle_indices = numpy.random.permutation(numpy.arange(len(y)))
#Shuffled value of x
x_shuffled = x[shuffle_indices]
#Shuffled value of y
y_shuffled = y[shuffle_indices]

#-->Split data into training and testing data.
#Testing/Validation data is dev_percent of data. In this case it is 10% of the data
index_dev = -1 * int(dev_percent * float(len(y)))
x_train, x_dev = x_shuffled[:index_dev], x_shuffled[index_dev:]
y_train, y_dev = y_shuffled[:index_dev], y_shuffled[index_dev:]

#-->Training
#Create the tensorflow graph and session
with tensorflow.Graph().as_default():
    #Define session configuration
    configuration = tensorflow.ConfigProto(allow_soft_placement=soft_pl,log_device_placement=log_pl)
    #Create the session using the configuration created above
    sess = tensorflow.Session(config=configuration)
    with sess.as_default():
        #Classify text using TextCNN from file TextConvNet. THis is the convolution neural network for text
        #classification
        conv_neural_net = TextConvNet.TextCNN(len_seq=x_train.shape[1],class_num=y_train.shape[1],
                                              vocab=len(process_voc.vocabulary_),sz_emb=embed_size,
                                              sz_fil=fil_sz,fil_num=fil_num,l2_reg_lambda=l2)

        # Define Training procedure
        #Define the global step and create variable tensor
        stp_glo = tensorflow.Variable(0, name="stp_glo", trainable=False)
        #Create optimizer with a learning rate of learning_rate
        opt = tensorflow.train.AdamOptimizer(learning_rate)
        #Determine gradients and variables
        gradient = opt.compute_gradients(conv_neural_net.loss)
        #Create training optimizer
        train_op = opt.apply_gradients(gradient, global_step=stp_glo)
        #Determine the values of gradient values and sparsity
        grad_summaries = []
        for g, v in gradient:
            if g is not None:
                #Write gradient history summary
                grad_hist_summary = tensorflow.summary.histogram("{}/grad/hist".format(v.name), g)
                #Write sparsity summary
                sparsity_summary = tensorflow.summary.scalar("{}/grad/sparsity".format(v.name),
                                                             tensorflow.nn.zero_fraction(g))
                #Add to grad_Summaries from gradient history and sparsity history
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        #Merge summary for grad_summaries
        grad_summaries_merged = tensorflow.summary.merge(grad_summaries)

        #Determine directory for output of the models and summary
        #timestamp
        timestamp = str(int(time.time()))
        #Output directory
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))

        #-->Loss Summary
        loss_summary = tensorflow.summary.scalar("loss", conv_neural_net.loss)
        #-->Accuracy Summary
        acc_summary = tensorflow.summary.scalar("accuracy", conv_neural_net.accuracy)

        #-->Train Summaries
        #Train summary optimizer
        train_summary_op = tensorflow.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        #Train summary directory
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        #Train summary writer
        train_summary_writer = tensorflow.summary.FileWriter(train_summary_dir, sess.graph)

        #-->Validation Set Summaries
        #Validation summary optimizer
        dev_summary_op = tensorflow.summary.merge([loss_summary, acc_summary])
        #Validation summary directory
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        #Validation summary writer
        dev_summary_writer = tensorflow.summary.FileWriter(dev_summary_dir, sess.graph)

        #Checkpoint directory within current directory
        #Directory for created checkpoints
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        #Check if directory exists. If yes, create a new one
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        #Save to tensorboard
        saver = tensorflow.train.Saver(tensorflow.global_variables(), max_to_keep=ckpt_num)

        #-->Write vocabulary
        process_voc.save(os.path.join(out_dir, "vocab"))

        #-->Initialize variables
        sess.run(tensorflow.global_variables_initializer())

        #-->Generate batches inside FileFuntion
        batches = FileFunction.iteration_bat(
            list(zip(x_train, y_train)), batch_sz, epoch_num)
        #-->Training loop.
        for batch in batches:
            #Separate batch into x_batch and y_
            x_batch, y_batch = zip(*batch)
            #Define the feed_dict to load into tensorflow placeholders for conv_neural_net for training
            feed_dict = {
                conv_neural_net.input_x: x_batch,
                conv_neural_net.input_y: y_batch,
                conv_neural_net.drop_keep: drop_keep
            }
            #Run the training iteration inside each batch. Returns the step, summaries, loss, and accuracy
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, stp_glo, train_summary_op, conv_neural_net.loss, conv_neural_net.accuracy],
                feed_dict)
            #Create a timestamp for each training iteration
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

            current_step = tensorflow.train.global_step(sess, stp_glo)

            #Validation part of training
            if current_step % eval_on_dev == 0:
                print("\nEvaluation:")
                # Define the feed_dict to load into tensorflow placeholders for conv_neural_net for validation
                feed_dict = {
                    conv_neural_net.input_x: x_batch,
                    conv_neural_net.input_y: y_batch,
                    conv_neural_net.drop_keep: 1.0
                }
                #Run the validation iteration inside each batch. Returns the step, summaries, loss, and accuracy
                step, summaries, loss, accuracy = sess.run(
                    [stp_glo, dev_summary_op, conv_neural_net.loss, conv_neural_net.accuracy],
                    feed_dict)
                # Create a timestamp for each training iteration
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                #Write validation summary
                if dev_summary_writer:
                    dev_summary_writer.add_summary(summaries, step)
                print("")
            #Save model after each checkpoint
            if current_step % ckpt == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))