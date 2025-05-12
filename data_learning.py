import torch
import torch.optim as optim
import torch.nn as nn

import model 
import utils_


class DataLearning:
    def __init__(self, dataloader_train, dataloader_test, num_epochs):
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test
        self.current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.Net()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(),lr=0.001)
        self.num_epochs = num_epochs
        self.batch_size = dataloader_train.batch_size
        self.status_report = utils_.StatusReport()

        self.dict = {}
        self.dict["dict_name"] = "DataLearning"
        self.dict["current_device"] = self.current_device
        self.dict["model"] = self.model.to(self.current_device)
        self.dict["criterion"] = self.criterion
        self.dict["optimizer"] = self.optimizer
        self.dict["num_epochs"] = self.num_epochs
        self.dict["batch_size"] = self.batch_size
        self.dict["train_batch_num"] = []
        self.dict["train_loss"] = []
        self.dict["train_acc_batch"] = []
        self.dict["train_acc_epoch"] = []
        self.dict["test_batch_num"] = []
        self.dict["test_loss"] = []
        self.dict["test_acc_batch"] = []
        self.dict["test_acc_epoch"] = []

    def get_dataloaders(self):
        return self.dataloader_train, self.dataloader_test, self.dataloader_validation 
    
    def train_model(self, epoch):
        self.status_report.headline("Train")
        total = 0
        correct = 0

        for batch_number, data_batch in enumerate(self.dataloader_train):
            self.optimizer.zero_grad()
            inputs = data_batch["img"].type(torch.float32).to(self.current_device)
            labels = data_batch["label"].to(self.current_device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total += labels.size(0)
            correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            acc = 100 * correct // total

            self.status_report.status(epoch=epoch,batch_num=batch_number,acc=acc,loss=loss.item(),printevery=100)
            self.dict["train_loss"].append(loss.item())
            self.dict["train_acc_batch"].append(acc)
            self.dict["train_batch_num"].append(batch_number)
        self.dict["train_acc_epoch"].append(acc)



    def test_model(self, epoch):
        self.status_report.headline("Test")
        total = 0
        correct = 0
        with torch.no_grad():
            for batch_number, data_batch in enumerate(self.dataloader_test):
                inputs = data_batch["img"].type(torch.float32).to(self.current_device)
                labels = data_batch["label"].to(self.current_device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                total += labels.size(0)
                correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
                acc = 100 * correct // total
                self.status_report.status(epoch=epoch,batch_num=batch_number,acc=acc,loss=loss.item(),printevery=100)
                self.dict["test_loss"].append(loss.item())
            self.dict["test_acc_batch"].append(acc)
            self.dict["test_batch_num"].append(batch_number)
        self.dict["test_acc_epoch"].append(acc)
            
    
    def evaluate_model(self):
        figlist = [utils_.graph_train_acc(self.dict),
                    utils_.graph_train_loss(self.dict),
                    utils_.graph_test_acc(self.dict),
                    utils_.graph_test_loss(self.dict),
        ]
        


    def save(self):
        torch.save(self.model, "model.pt")
        torch.save(self.dict, "dict.pt")
        
    def routine(self):
        epoch_list = []
        for epoch in range(self.num_epochs):
            self.train_model(epoch)
            self.save()
            self.test_model(epoch)
            self.evaluate_model()
            
            epoch_list.append(epoch)
            self.dict["current_epoch"] = epoch

    def write_log(self, dict_):
        file = open("log.txt", "w")  # open file in write mode
        file.write("{\n")   
        for key in dict_.keys():        
            file.write(f"'{key}': '{dict_[key]}',\n")  # add comma at end of line
        file.write("}")
        file.close()    




if __name__ == "__main__":
    import torch
    import torch.optim as optim
    import torch.nn as nn
    import numpy as np

    import model 
    from datasets import Dataset 

    data = np.random.rand(16,10,3,32,32)
    label = np.random.randint(0, 10, size=16)
    ds = Dataset.from_dict({"data": data, "label": label}).with_format("torch")
    dataloader_train = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=2)
    dataloader_test = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=2)
    dataloader_validation = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=2)

    for batch_number, batch in enumerate(dataloader_train, 0):
        input = batch["data"]
        label = batch["label"]
        print(label)

    data_learning = DataLearning(dataloader_train, dataloader_test, dataloader_validation, num_epochs=5)
    data_learning.routine()
    data_learning.write_log(data_learning.dict)
