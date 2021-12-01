from utils.model import Perceptron
import pandas as pd
import numpy as np
from utils.all_utils import prepare_data, save_plot, save_model
import logging 
import os

logging_str = "[%(asctime)s:%(levelname)s:%(module)s] %(message)s"
log_dir="logs"
os.makedirs(log_dir, exist_ok=True)
#%(asctime)s: =>Time in which code is executed 
#%(levelname)s: => Whcih information are you getting ..Is it the debug information ..or error information..or exception information etc
#%(module)s: =>Which module has raised the error information
#%(message)s: =>What message to print..eg:if the training is started ...if there is any exceptions 
logging.basicConfig(filename=os.path.join(log_dir,"running_logs.log"),level=logging.INFO,format=logging_str,filemode='a')
#filemode='a'used to append all the logs after each run
#basic configuration is set only in the main file. We are calling all the modules in here.

def main(data,eta,epochs,filename,plotFileName):
    

    df = pd.DataFrame(data)
    logging.info(f"This is the actual dataframe {df}")

    X,y = prepare_data(df)


    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)

    _ = model.total_loss()

    save_model(model,filename)
    save_plot(df,plotFileName, model)

if __name__=='__main__':

    OR = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [0,1,1,1],
    }

    ETA = 0.3 # 0 and 1
    EPOCHS = 10

    try:
        logging.info(">>>>> Starting training >>>>>")
        main(data=OR,eta=ETA,epochs=EPOCHS,filename="or.model",plotFileName="or.png")
        logging.info("<<<<< Training doen successfully <<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e