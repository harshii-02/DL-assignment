# DL-assignment
1.Alexnet
The original AlexNet was modified to create an optimized version by reducing the number of filters, adding Batch Normalization, and using He Normal initialization for better training stability. Large fully connected layers and the Flatten layer were replaced with Global Average Pooling and a smaller dense layer to reduce parameters and overfitting. Padding was changed to preserve spatial dimensions, the number of output classes was reduced to suit a smaller dataset, and the model was compiled and tested. Overall, the optimized model is more efficient, faster to train, and better suited for modern CNN training.

2.Cat and Dog
The code was simplified by using image_dataset_from_directory instead of manual data loading and ImageDataGenerator. Image preprocessing was moved into the model using a Rescaling layer. The CNN architecture was made simpler, the optimizer was changed to Adam, batch size was increased, and training epochs were reduced. Overall, the updated code is cleaner, more modern, and easier to maintain.

3.Deepreinforcement
The code was optimized by removing redundancy, simplifying Q-learning updates, replacing deprecated NumPy matrices, improving readability, and streamlining training and testing while preserving the original reinforcement learning logic.

4.Lstm
The original code implements a Q-learning based reinforcement learning algorithm for optimal path finding in a graph environment. The modified code replaces this approach with an LSTM-based deep learning model for time-series forecasting of airline passenger data, introducing dataset handling, normalization, sequence modeling, neural network training, and regression-based evaluation
