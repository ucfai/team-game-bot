from model import Model
import datetime

def save_model(model, argv):
    name_of_model = argv[1] if len(argv) == 2 else "Model__" + str(datetime.datetime.now())[:-7].replace(" ", "__")

    print("Saving trained model to models/{}".format(name_of_model))
    model.save_to('models/{}'.format(name_of_model))