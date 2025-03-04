from typing import Dict, Optional, Tuple, List, Union, Callable
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.common import (
    Metrics, EvaluateIns, EvaluateRes, FitIns, FitRes, MetricsAggregationFn, 
    Scalar, logger, Parameters, NDArray, NDArrays
)
from flwr.server.strategy.aggregate import weighted_loss_avg
from logging import WARNING
from src.core.util.mapper import *
import flwr as fl

from src.nn.layers.layer_util import aggregate_custom
from src.federated_learning.federation.aggregate import aggregate


class FedCustom(fl.server.strategy.Strategy):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.fraction_fit = kwargs.get("fraction_fit", None)
        self.fraction_evaluate = kwargs.get("fraction_evaluate", None)
        self.min_fit_clients = kwargs.get("min_fit_clients", None)
        self.min_evaluate_clients = kwargs.get("min_evaluate_clients", None)
        self.min_available_clients = kwargs.get("min_available_clients", None)
        self.evaluate_metrics_aggregation_fn = kwargs.get("evaluate_metrics_aggregation_fn", None)
        self.initial_parameters = kwargs.get("initial_parameters", None)
        self.fit_metrics_aggregation_fn = kwargs.get("fit_metrics_aggregation_fn", None)
        self.evaluate_fn = kwargs.get("evaluate_fn", None)
        self.on_fit_config_fn = kwargs.get("on_fit_config_fn", None)
        self.on_evaluate_config_fn = kwargs.get("on_evaluate_config_fn", None)
        self.accept_failures = kwargs.get("accept_failures", True)
        self.context_server = kwargs.get("context_server", None)
        self.context_client = kwargs.get("context_client", None)
        self.lr = kwargs.get("lr", None)
        self.model_save = kwargs.get("model_save", None)
        self.central_model = kwargs.get("central_model", None)
        self.test_loader = kwargs.get("test_loader", None)
        self.device = kwargs.get("device", "cpu")  # Default to "cpu"
        self.checkpoint = kwargs.get("checkpoint", None)
        self.path_crypted = kwargs.get("path_crypted", None)
    
    def __repr__(self) -> str:
        # Same function as FedAvg(Strategy)
        return f"FedCustom (accept_failures={self.accept_failures})"

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        # Same function as FedAvg(Strategy)
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        # Same function as FedAvg(Strategy)
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        
        # Create custom configs
        n_clients = len(clients)
        # Custom fit config function provided
        standard_lr = self.lr
        config = {"server_round": server_round, "local_epochs": 1}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)

        # fit_ins = FitIns(parameters, config)
        # Return client/config pairs
        fit_configurations = []
        for idx, client in enumerate(clients):
            config["learning_rate"] =  standard_lr #standard_lr if idx < half_clients else higher_lr
            """
            Each pair of (ClientProxy, FitRes) constitutes 
            a successful update from one of the previously selected clients.
            """
            fit_configurations.append(
                (
                    client,
                    FitIns(
                        parameters,
                        config
                    )
                )
            )
        # Successful updates from the previously selected and configured clients
        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average. (each round)"""
        # Same function as FedAvg(Strategy)
        if not results:
            return None, {}

        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results parameters --> array matrix
        weights_results = [
            (parameters_to_ndarrays_custom(fit_res.parameters, self.context_server), fit_res.num_examples)
            for _, fit_res in results
        ]

        # Aggregate parameters using weighted average between the clients and convert back to parameters object (bytes)
        parameters_aggregated = ndarrays_to_parameters_custom(aggregate_custom(weights_results))

        metrics_aggregated = {}
        # Aggregate custom metrics if aggregation fn was provided
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        elif server_round == 1:  # Only log this warning once
            logger.log(WARNING, "No fit_metrics_aggregation_fn provided")

        # Same function as SaveModelStrategy(fl.server.strategy.FedAvg)
        """Aggregate model weights using weighted average and store checkpoint"""
        aggregate(server_round, parameters_aggregated, 
                  self.central_model, self.checkpoint, self.context_client,
                  self.context_server, self.path_crypted)
        return parameters_aggregated, metrics_aggregated

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        # Same function as FedAvg(Strategy)
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Same function as FedAvg(Strategy)
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}  # {"server_round": server_round, "local_epochs": 1}

        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )

        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        # Each pair of (ClientProxy, FitRes) constitutes a successful update from one of the previously selected clients
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        # Same function as FedAvg(Strategy)
        if not results:
            return None, {}

        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        metrics_aggregated = {}
        # Aggregate custom metrics if aggregation fn was provided
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)

        # Only log this warning once
        elif server_round == 1:
            logger.log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters using an evaluation function."""
        # Same function as FedAvg(Strategy)
        if self.evaluate_fn is None:
            # Let's assume we won't perform the global model evaluation on the server side.
            return None

        # if we have a global model evaluation on the server side :
        parameters_ndarrays = parameters_to_ndarrays_custom(parameters, self.context_server)
        eval_res = self.evaluate_fn(server_round=server_round, parameters=parameters_ndarrays, 
                                    config={}, testloader=self.test_loader, 
                                    device=self.device, model_init=self.central_model)

        # if you haven't results
        if eval_res is None:
            return None

        loss, metrics = eval_res
        return loss, metrics