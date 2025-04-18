import os

from data_acquisition import DataAcquisition
from data_preprocession import DataTransformation
from data_learning import DataLearning

os.environ['KMP_DUPLICATE_LIB_OK']='True'

#data acquisition and transformation
data_link_huggingface = "uoft-cs/cifar10"
data_acquisition = DataAcquisition(data_path="test",
                                data_name="test")
data_transformation = DataTransformation()
dataset_train = data_acquisition.load_data_huggingface(data_link_huggingface, split = "train").with_transform(data_transformation.transforms_torch_for_huggingface)
dataset_test = data_acquisition.load_data_huggingface(data_link_huggingface, split = "test").with_transform(data_transformation.transforms_torch_for_huggingface)
dataloader_train = data_transformation.dataset_to_dataloader(dataset_train, batch_size=128)
dataloader_test = data_transformation.dataset_to_dataloader(dataset_test, batch_size=128)

#data learning and evaluation
data_learning = DataLearning(dataloader_train, dataloader_test, num_epochs=4)
data_learning.routine()
data_learning.write_log(data_learning.dict)


#BUAT NEW FILE TIAP RUN
