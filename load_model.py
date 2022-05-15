from sklearn.metrics import classification_report

from train_model import y_test, create_model, X_test_pad

model = create_model()
model.load_weights('checkpoints/sarcasm_checkpoint')
test_prediction = [True if i[0] * 100 > 50 else False for i in model.predict(X_test_pad)]

print(classification_report(y_test, test_prediction, target_names=['Not Sarcastic', 'Sarcastic']))
