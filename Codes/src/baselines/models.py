from src.baselines.RCGAN import RCGANTrainer
from src.baselines.networks.discriminators import LSTMDiscriminator
from src.baselines.networks.generators import LSTMGenerator
import torch
from src.evaluations.test_metrics import get_standard_test_metrics
from src.utils import loader_to_tensor, loader_to_cond_tensor
GENERATORS = {'LSTM': LSTMGenerator}


def get_generator(generator_type, input_dim, output_dim, **kwargs):
    return GENERATORS[generator_type](input_dim=input_dim, output_dim=output_dim, **kwargs)


DISCRIMINATORS = {'LSTM': LSTMDiscriminator}


def get_discriminator(discriminator_type, input_dim, **kwargs):
    return DISCRIMINATORS[discriminator_type](input_dim=input_dim, **kwargs)


def get_trainer(config, train_dl, test_dl):
    # print(config)

    print(config.gan_algo)
    if config.dataset == "BerkeleyMHAD":
        model_name = "%s_%s" % (
            config.dataset, config.gan_algo)

    else:
        model_name = "%s_%s" % (config.dataset, config.gan_algo)
    if config.conditional:
        config.update({"G_input_dim": config.G_input_dim +
                      config.num_classes}, allow_val_change=True)
        x_real_train = torch.cat([loader_to_tensor(
            train_dl), loader_to_cond_tensor(train_dl, config)], dim=-1).to(config.device)
        x_real_test = torch.cat([loader_to_tensor(
            test_dl), loader_to_cond_tensor(test_dl, config)], dim=-1).to(config.device)
    else:
        x_real_train = loader_to_tensor(
            train_dl).to(config.device)
        x_real_test = loader_to_tensor(
            test_dl).to(config.device)

    print(model_name)
    if config.gan_algo == 'PathChar_GAN':
        D_out_dim = config.D_out_dim
    else:
        D_out_dim = 1

    generator = GENERATORS[config.generator](
        input_dim=config.G_input_dim, hidden_dim=config.G_hidden_dim, output_dim=config.input_dim, n_layers=config.G_num_layers)
    discriminator = DISCRIMINATORS[config.discriminator](
        input_dim=config.input_dim, hidden_dim=config.D_hidden_dim, out_dim=D_out_dim, n_layers=config.D_num_layers)
    print('GENERATOR:', generator)
    print('DISCRIMINATOR:', discriminator)
    # Compute test metrics for train and test set
    test_metrics_train = get_standard_test_metrics(x_real_train)
    test_metrics_test = get_standard_test_metrics(x_real_test)
    trainer = {
        "ROUGH_RCGAN": RCGANTrainer(G=generator, D=discriminator,
                                    test_metrics_train=test_metrics_train, test_metrics_test=test_metrics_test,
                                    train_dl=train_dl, batch_size=config.batch_size, n_gradient_steps=config.steps,
                                    config=config)}[model_name]

    # Check if multi-GPU available and if so, use the available GPU's
    print("GPU's available:", torch.cuda.device_count())
    # Required for multi-GPU
    torch.backends.cudnn.benchmark = True

    return trainer
