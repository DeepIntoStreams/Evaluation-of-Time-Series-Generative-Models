from eval_helper import *

def get_discriminative_score(real_train_dl, real_test_dl, fake_train_dl, fake_test_dl, config):
    
    class Discriminator(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, out_size=2):
            super(Discriminator, self).__init__()
            self.rnn = nn.GRU(input_size=input_size, num_layers=num_layers,
                                hidden_size=hidden_size, batch_first=True)
            self.linear = nn.Linear(hidden_size, out_size)

        def forward(self, x):
            x = self.rnn(x)[0][:, -1]
            return self.linear(x)

    train_dl = create_dl(real_train_dl, fake_train_dl, config.batch_size)
    test_dl = create_dl(real_test_dl, fake_test_dl, config.batch_size)

    pm = TrainValidateTestModel(epoch=config.epoch,device=config.device)
    test_acc_list = []
    for i in range(1):
        model = Discriminator(train_dl.dataset[0][0].shape[-1], config.hidden_size, config.num_layers)
        _, _, test_acc = pm.test_classification(train_dl,test_dl,model,train=True,validate=True)
        test_acc_list.append(test_acc)
    mean_acc = np.mean(np.array(test_acc_list))
    std_acc = np.std(np.array(test_acc_list))
    return abs(mean_acc-0.5), std_acc

def compute_predictive_score(real_train_dl, real_test_dl, fake_train_dl, fake_test_dl, config):

    train_dl = create_dl(fake_train_dl, fake_test_dl, config.batch_size, cutoff=True)
    test_dl = create_dl(real_train_dl, real_test_dl, config.batch_size, cutoff=True)

    class Predictor(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, out_size):
            super(Predictor, self).__init__()
            self.rnn = nn.LSTM(input_size=input_size, num_layers=num_layers,
                               hidden_size=hidden_size, batch_first=True)
            self.linear = nn.Linear(hidden_size, out_size)

        def forward(self, x):
            x = self.rnn(x)[0][:, -1]
            return self.linear(x)

    pm = TrainValidateTestModel(epoch=config.epochs,device=config.device)
    test_loss_list = []
    for i in range(1): ## Question: WHY 1?
        model = Predictor(train_dl.dataset[0][0].shape[-1], 
                          config.hidden_size, 
                          config.num_layers, 
                          out_size=train_dl.dataset[0][1].shape[-1]
                          )
        model, test_loss = pm.test_regressor(
                        train_dl=train_dl, 
                        test_dl=test_dl, 
                        model = model, 
                        train=True, 
                        validate=True
                        ) 
        test_loss_list.append(test_loss)
    mean_loss = np.mean(np.array(test_loss_list))
    std_loss = np.std(np.array(test_loss_list))
    return mean_loss, std_loss

