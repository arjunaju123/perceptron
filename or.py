from utils.model import Perceptron
import pandas as pd
import numpy as np
from utils.all_utils import prepare_data, save_plot, save_model

def main(data,eta,epochs,filename,plotFileName):
    

    df = pd.DataFrame(data)


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

    main(data=OR,eta=ETA,epochs=EPOCHS,filename="or.model",plotFileName="or.png")