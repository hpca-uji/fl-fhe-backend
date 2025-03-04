from src.core.util.common import *
from src.nn.engine.test import test
from src.nn.engine.train import train
from flwr.client import NumPyClient
from example.models.custom_model import Net
from src.core.util.common import check_directory
from datetime import datetime

class FlowerClient(NumPyClient):
    def __init__(self, 
                 cid, 
                 net, 
                 trainloader, 
                 valloader, 
                 device, 
                 batch_size, 
                 save_results, 
                 plot_path, 
                 roc_path,
                 yaml_path, 
                 he, 
                 classes, 
                 context_client):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.batch_size = batch_size
        self.save_results = save_results
        self.plot_path = plot_path
        self.roc_path = roc_path
        self.yaml_path = yaml_path
        self.he = he
        self.classes = classes
        self.context_client = context_client
        
        print(self.device)

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters2(self.net, self.context_client)

    def fit(self, parameters, config):
        server_round = config['server_round']
        local_epochs = config['local_epochs']
        lr = float(config["learning_rate"])

        print(f'[Client {self.cid}, round {server_round}] fit, config: {config}')

        set_parameters(self.net, parameters, self.context_client)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        results = train(self.net, self.trainloader, self.valloader, optimizer=optimizer, loss_fn=criterion,
                               epochs=local_epochs, device=self.device)

        if self.save_results:
            save_graphs(self.plot_path, local_epochs, results, f"_Client {self.cid}")

        return get_parameters2(self.net, self.context_client), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        set_parameters(self.net, parameters, self.context_client)

        loss, accuracy, y_pred, y_true, y_proba = test(self.net, self.valloader,
                                                              loss_fn=torch.nn.CrossEntropyLoss(), device=self.device)

        if self.save_results:
            current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            filename = f"client_{self.cid}"
            if self.plot_path:
                matrix_filename = os.path.join(self.plot_path, "matrix")
                check_directory(matrix_filename)
                save_matrix(y_true, y_pred, os.path.join(matrix_filename, filename ), self.classes)
            if self.roc_path:
                roc_filename = os.path.join(self.roc_path, "roc")
                check_directory(roc_filename)
                save_roc(y_true, y_proba, os.path.join(roc_filename, filename ), len(self.classes))

        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}