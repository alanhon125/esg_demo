import matplotlib.pyplot as plt
import json
file = '/model/layoutlm/layoutlm_large_500k_epoch_1/trainer_state.docparse_json'
with open(file, 'r', encoding='utf8') as f:
    history = json.load(f)['log_history'][:-1]
loss_train = [i['loss'] for i in history]
epochs = [i['epoch'] for i in history]
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()