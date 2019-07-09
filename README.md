# MachineLearningSeminar
This is a Machine Learning group project of Steven Becker
(steven.becker@tu-dortmund.de) and Felix Geyer (felix.geyer@tu-dortmund.de). We
created this framework as a task in the **Machine Learning Seminar** at **TU Dortmund**. Our
goal is to classify dog races based on roundabout 20,000 pictures of 120 dog
races.

The dataset was taken from this
[Website](http://vision.stanford.edu/aditya86/ImageNetDogs/). During the course
of the project, we created a smaller dataset containing only five races, in
particular chihuahua, beagle, schipperke, standard poodle and the African
hunting dog.

The framework `dog_classifier` can be installed with `pip install -e .` along
with all the required python libraries in a correct version (found in the
`requirements.txt`). In the `scripts` folder, executable scripts are found
which use certain parts of the framework.

## How to run the classifier

1. To create a **dataset**, use `python generate_dataset.py`
2. To create an **encoder-model**, use `python generate_encoder_model.py`. This is neccessary,
because one needs to transform the race_labels as numbers.
3. Training:
    1. To train a Neural Network architecture specified in `dog_classifier/net/network.py`,
    use `python training.py`.
    2. To train a Random Forest
