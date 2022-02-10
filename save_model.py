from model import Model
import datetime

def save_model(model, argv):
    model_name = name(argv)

    print("Saving trained model to models/{}".format(model_name))
    model.save_to('models/{}'.format(model_name))

def name(argv):
    return argv[1] if len(argv) == 2 else "Model__" + str(datetime.datetime.now())[:-7].replace(" ", "__")