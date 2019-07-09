# MachineLearningSeminar
This is a Machine Learning group project of Steven Becker
(steven.becker@tu-dortmund.de) and Felix Geyer (felix.geyer@tu-dortmund.de). We
created in as a task in the **Machine Learning Seminar** at **TU Dortmund**. Our
goal is to classify dog races based on roundabout 20,000 pictures of 120 dog
races.

The dataset was taken from this
[Website](http://vision.stanford.edu/aditya86/ImageNetDogs/). During the course
of the project, we created a smaller dataset containing only five races, in
particular chihuahua, beagle, schipperke, standard poodle and the African
hunting dog.

The framework `dog_classifier` can be installed with `pip install -e .` along
with all the required python libraries in an correct version (found in the
`requirements.txt`). In the `scripts` folder, executable scripts are found
which use certain parts of the framework.

The pipeline looks as follows:

1. To create a **dataset**, use `generate_dataset.py`
2. To create an **encoder-model**, use `generate_encoder_model.py`
