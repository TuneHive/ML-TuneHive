# ML-TuneHive

Repository for ML Development

**DISCLAIMER: Create different virtual environment for each project**

## Project Overview
Data used:

- [Spotify Music Streaming dataset in 4 years](https://www.kaggle.com/datasets/thedevastator/streaming-activity-dataset)

This Repository contains two projects. The first one is the model-dev that shows how the model was developed, and rest-api that was responsible for creating ML model deployment.

Inside the model-dev, The EDA process was done in EDA notebook inside the 'eda' directory, data engineering was done in data engineering notebook, and data preprocessing & modeling was done in preprocessing-modeling notebook.

The rest-api only implements single endpoint that handle the songs recommendation based on a songs sequence.

----

## How to train the model

1. To train the model to output relatively same results, we need to download the csv dataset in the notebook
2. Preprocess the csv dataset to tensorflow dataset using the `create_session_dataset_tensor` method
3. Run the definition of `GRU4REC` class, and function used in training and evaluation
4. Train the model. Because the data used here is small, the training process would not take a long time. It's advised to use the `train_gru4rec_with_strategy` to speed up the training process
5. used the `compute_recall_at_k` function to calculate the Recall@k of the model.
6. plot the training loss history using the `plot_traning_history` function
7. Save and load model using the code given at the end of notebook section


