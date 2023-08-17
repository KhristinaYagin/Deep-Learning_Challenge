                        Report on the Neural Network Model
* Overview of the Report:
    The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With the knowledge of machine learning and neural networks, use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup. From Alphabet Soup’s business team, you have received a CSV file from the organizations that have received funding from Alphabet Soup over the years. Within this dataset are numbers of columns that capture metadata about each organization's information. The dataset used for this analysis contains labeled instances, and the objective is to train a neural network to learn the underlying patterns in the data and make accurate predictions.

* Results of the Report:
    I. Data Preprocessing
    > What variable(s) are the target of the model?
    > What variable(s) are the features of the model?
    > What variable(s) should be removed from the input data because they are neither targets nor features?

    II. Compiling, Training, and Evaluating the Model
    > Neurons, Layers, and Activation Functions:
     The neural network model consists of three layers: an input layer, one or more hidden layers, and an output layer. The number of neurons in each layer and the choice of activation functions were as follows:
    - Input Layer: Number of neurons = Number of input features, Activation function = ReLU
    - Hidden Layer 1: Number of neurons = 80, Activation function = ReLU
    - Hidden Layer 2: Number of neurons = 30, Activation function = ReLU
    - Output Layer: Unit/s = 1, Activation function = Sigmoid
        The selection of these layers, neurons, and activation functions aims to capture complex relationships in the data and facilitate convergence during training.

    > Were you able to achieve the target model performance - Yes
    > What steps did you take in your attempts to increase model performance-             
        Hyperparameter Tuning: Various hyperparameters, such as learning rate, batch size, and the number of neurons, were tuned using techniques like grid search and random search.
        Regularization Techniques: Dropout layers were added to mitigate overfitting, and regularization was applied to the hidden layers and data augmentation techniques were employed to create additional training samples and improve the model's generalization.

* Report Summary
    In summary, the deep learning neural network model achieved the target performance of 78% accuracy on the validation set. This demonstrates its ability to accurately classify instances into the appropriate classes based on the provided features. The model architecture and hyperparameters were selected through iterative experimentation to optimize performance.

> Recommendation:
    For this classification problem, a different model approach could involve using a algorithm in handling tabular data and can handle feature interactions effectively. Unlike neural networks. Ultimately, the choice between a neural network depends on factors such as data size, complexity, interpretability needs, and available computational resources.



