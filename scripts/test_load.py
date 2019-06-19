from keras.models import load_model
import os

pfad = "build"
if not os.path.exists(pfad):
    os.makedirs(pfad)

model = load_model('build/my_model.h5')
print(model.summary())
