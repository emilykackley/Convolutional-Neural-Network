# Convolutional-Neural-Network

## CNN for Text Classification
  ### Approach
  - Load all data and parameters and prepare data for text classification
  - Build the vocabulary and determine the maximum length of the text document by splitting it into words from teh combined data of positive and negative samples. 
      * Process the vocabulary using the learn library from tensorflow.contrib and use the VocabularyProcessor using the maximum length of the text document
  - Split data into testing and training data
  - Begin training the model with a tensorflow graph and session.
  - Create a CNN model and begin training
  - Test model by using the validation set
  ### Limitations
  - This implementation is limited to a fixed learning rate. Ideally, the learning rate should be adapting in each iteration
  - Convolutional Neural Networks have a high computational cost and are slow to train without a good GPU and require a lot of training data
  - Tensorflow can take a lot of GPU resources
      
