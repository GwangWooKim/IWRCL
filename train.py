from utils import *

from tqdm import tqdm
import json
import matplotlib.pyplot as plt
# import arviz as az
# az.style.use("arviz-darkgrid")

def train(train_loader, model, optimizer):

    model.train()

    if model.model == "VAE":
        def model_forward(x):
            return model.VAE_forward(x)
    elif model.model == "IWAE":
        def model_forward(x):
            return model.IWAE_forward(x)
    elif model.model == "IWRCL":
        def model_forward(x):
            return model.IWRCL_forward(x)
    else:
        raise ValueError(f"Unsupported model type: {model.model}")
    
    ll_lst, kl_lst, loss_lst = [], [], []
    for epoch in tqdm(range(model.epochs), desc="epoch"):
        sum_ll = torch.zeros((), device=model.device)
        sum_kl = torch.zeros((), device=model.device)
        sum_loss = torch.zeros((), device=model.device)
        n_samples = 0

        for step, x in enumerate(train_loader):
            # training
            x = x.to(model.device)
            optimizer.zero_grad()

            ll, kl, loss = model_forward(x)
            loss.backward()
            optimizer.step()

            # logging
            bs = x.size(0)

            sum_ll += ll.detach() * bs
            sum_kl += kl.detach() * bs
            sum_loss += loss.detach() * bs
            n_samples += bs
            
        ll_lst.append((sum_ll / n_samples).item())
        kl_lst.append((sum_kl / n_samples).item())
        loss_lst.append((sum_loss / n_samples).item())

    model.train_log = {
        "ll" : ll_lst,
        "kl" : kl_lst,
        "loss" : loss_lst,
    }
    
    return model

def save(model, train_loader, path = None, plot = True):
    seed_everything(model.seed)

    if path is None:
        path = "./save/last"
    os.makedirs(path, exist_ok=True)
    
    with torch.inference_mode():
        if model.data_name == "cortex":
            # data encoding
            x = train_loader.dataset.to(model.device)
            mu, _ = model.encoder(x)
            generated = mu.squeeze(1).cpu().numpy()
            
        else:
            # data generation
            if model.data_name == "VerbAgg":
                N = 10000
            else:
                N = train_loader.dataset.size(0)

            z = torch.randn(N, model.z_dim).to(model.device)
            generated = model.decoder(z).cpu().numpy()

            # postprocessing
            if train_loader.std:
                generated = generated * train_loader.std + train_loader.mean
        
    plt.figure()
    plt.plot(model.train_log["ll"], label = "ll")
    plt.plot(model.train_log["kl"], label = "kl")
    plt.plot(model.train_log["loss"], label = "loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{path}/train_log.png")
    if plot:
        plt.show()
    plt.close()
    
    with open(f"{path}/kwargs.json", "w", encoding="utf-8") as f:
        kwargs = dict(model.kwargs)
        kwargs["device"] = str(model.kwargs["device"])
        json.dump(kwargs, f, indent=2)

    np.save(f"{path}/generated.npy", generated)
    torch.save(model.state_dict(), f"{path}/model_states.pt")
    