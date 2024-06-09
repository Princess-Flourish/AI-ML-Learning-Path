import pickle

# Save the pipeline using pickle
pickle.dump(pipeline, open(filename, 'wb'))

# Open the saved file 'trained_model.sav' in binary read mode ('rb')
loaded_model = pickle.load(open('C:/Users/Mandela Tangban/Documents/Deploying Machine learning mode/trained_model.sav', 'rb'))