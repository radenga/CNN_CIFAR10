import time
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np

class StatusReport():
    def __init__(self):
        """
        Usage : 
        epoch = 1
        batch_num = 100
        acc = 1
        loss = 0.005
        mode = "Train"

        status_report = StatusReport()
        status_report.headline(mode)
        for batch_num in range(batch_num):
            status_report.status(epoch,batch_num,acc,loss)
            time.sleep(0.2)
        """
        a=1
    def headline(self, mode):
        print("\n \t {} Mode".format(str(mode)))
        print("Epoch\tBatch\tAcc\t{}_Loss\t".format(str(mode)), end="\n")
    def status(self,
              epoch,
              batch_num,
              acc,
              loss,
              printevery=10,):
        print("\r{}\t{}\t{}\t{}".format(epoch,batch_num,acc,round(loss,5)), end="")
        if batch_num % printevery == 0:
            print("\n",end="")



class FolderCreator():
    def __init__(self):
        self.now = datetime.now()
        self.datetime_str = self.now.strftime("%Y_%m_%d__%H_%M_%S") 
        self.cwd = os.getcwd()
        self.base_wd = self.cwd+"\\Result\\"+self.datetime_str 
        a=1
    def create_file(self, input_dirs="0"):
        if input_dirs == "0":
            self.wd = self.base_wd 
            if not os.path.exists(self.base_wd):
                os.makedirs(self.base_wd)
        else:
            self.wd = self.cwd+"\\Result\\"+input_dirs
            if not os.path.exists(self.wd):
                os.makedirs(self.wd)



#&& Graph
def TrainLossGraph(dict):
    y = dict["train_loss"]
    x = list(range(dict["train_loss"]))
    plt.plot(x,y)
    plt.show()
    plt.savefig("train_loss")
