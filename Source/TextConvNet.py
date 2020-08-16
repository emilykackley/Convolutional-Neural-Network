import tensorflow

class TextCNN(object):
    # len_seq = length of our sentences,classes = number of classes in output layer, vocab = size of vocabulary (needed
    # to define size of embedding layer), embed_sz = dimensionality of embeddings, fil_sz = number of words
    # convolutional filters cover, filters = number of filters per filter size
    def __init__(
      self, len_seq, class_num, vocab,sz_emb, sz_fil, fil_num, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tensorflow.placeholder(tensorflow.int32, [None, len_seq], name="input_x")
        self.input_y = tensorflow.placeholder(tensorflow.float32, [None, class_num], name="input_y")
        self.drop_keep = tensorflow.placeholder(tensorflow.float32, name="drop_keep")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tensorflow.constant(0.0)

        # Define embedding layer that maps vocabulary word indices into low-dim vectors
        # This is a lookup table learned from the data

        # Forces an operation to be executed on CPU instead of the default GPU if available
        # Embeddings does not support GPU execution so CPU is needed to execute
        with tensorflow.device('/cpu:0'), tensorflow.name_scope("embedding"):
            # Embedding matrix learned at training. tensorflow.embedding_lookup creates the embedding operation
            # Result will be a 3-D tensor with shape[None, len_seq, sz_emb]
            self.W = tensorflow.Variable(tensorflow.random_uniform([vocab, sz_emb], -1.0, 1.0),name="W")
            self.emb_ch = tensorflow.nn.embedding_lookup(self.W, self.input_x)
            self.emb_ch_exp = tensorflow.expand_dims(self.emb_ch, -1)

        # Build convolutional layers followed by max-pooling
        # Each convolution produces tensors with different shapes. Layers will be created for each of them
        #  then combined into a feature vector

        outs = []
        for i, filter_size in enumerate(sz_fil):
            with tensorflow.name_scope("conv-maxpool-%s" % filter_size):
                # Layer for Convolution
                shape_fil = [filter_size, sz_emb, 1, fil_num]
                W = tensorflow.Variable(tensorflow.truncated_normal(shape_fil, stddev=0.1), name="W")
                b = tensorflow.Variable(tensorflow.constant(0.1, shape=[fil_num]), name="b")
                convolution = tensorflow.nn.conv2d(self.emb_ch_exp,W,strides=[1, 1, 1, 1],padding="VALID",
                                            name="conv")
                # Nonlinearity Application
                h = tensorflow.nn.relu(tensorflow.nn.bias_add(convolution, b), name="relu")
                # Maxpooling over the outputs
                pd = tensorflow.nn.max_pool(h,ksize=[1, len_seq - filter_size + 1, 1, 1],strides=[1, 1, 1, 1],
                                                padding='VALID',name="pool")
                outs.append(pd)

        #Combine into feature vector
        num_filters_total = fil_num * len(sz_fil)
        self.h_pool = tensorflow.concat(outs, 3)
        self.h_pool_flat = tensorflow.reshape(self.h_pool, [-1, num_filters_total])

        # Dropout Layer
        # Dropout later desables a fraction of its neurons. This prevents neurons from co-adapting
        # and forces them to learn individually useful features

        # Add the dropout layer
        with tensorflow.name_scope("dropout"):
            self.h_drop = tensorflow.nn.dropout(self.h_pool_flat, self.drop_keep)

        # Determine prediction and score from feature vector from max pooling with the dropout applied
        # Complete matrix multiplication and pick the class with highest score
        with tensorflow.name_scope("output"):
            W = tensorflow.get_variable("W",shape=[num_filters_total, class_num],
                                        initializer=tensorflow.contrib.layers.xavier_initializer())
            b = tensorflow.Variable(tensorflow.constant(0.1, shape=[class_num]), name="b")
            l2_loss += tensorflow.nn.l2_loss(W)
            l2_loss += tensorflow.nn.l2_loss(b)
            self.scores = tensorflow.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tensorflow.argmax(self.scores, 1, name="predictions")

        # Calculate the loss
        with tensorflow.name_scope("loss"):
            losses = tensorflow.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
            self.loss = tensorflow.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Calculate the accuracy
        with tensorflow.name_scope("accuracy"):
            correct_predictions = tensorflow.equal(self.predictions, tensorflow.argmax(self.input_y, 1))
            self.accuracy = tensorflow.reduce_mean(tensorflow.cast(correct_predictions, "float"), name="accuracy")
