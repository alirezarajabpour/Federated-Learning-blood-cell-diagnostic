import flwr as fl
from flwr.common import FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pickle
import mlflow


def update_hpm_state_dict(hpm_state, new_model_state, beta):
    return [beta * old + (1 - beta) * new for old, new in zip(hpm_state, new_model_state) ]


class MapStrategy(fl.server.strategy.FedAvg):
    def __init__(self, hpm_beta, initial_parameters: Parameters, **kwargs):
        super().__init__(initial_parameters=initial_parameters, **kwargs)
        self.hpm_beta = hpm_beta
        self.client_hpms: Dict[str, List[np.ndarray]] = {}
        self.final_parameters: Optional[Parameters] = initial_parameters

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        """Configure training. The client's logical ID is not known here."""
        fit_ins_list = super().configure_fit(server_round, parameters, client_manager)

        return fit_ins_list

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, fl.common.EvaluateIns]]:
        """Configure evaluation. The client's logical ID is not known here."""

        return super().configure_evaluate(server_round, parameters, client_manager)

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results and update client HPMs."""
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            self.final_parameters = aggregated_parameters

        for client_proxy, fit_res in results:
            if "logical_id" not in fit_res.metrics:
                continue

            logical_cid = str(fit_res.metrics.pop("logical_id"))

            if server_round == 1 and "train_samples" in fit_res.metrics:
                train_samples = fit_res.metrics.pop("train_samples")
                mlflow.log_metric(f"client_{logical_cid}_train_samples", train_samples, step=0)

            if "p_k_t_state_dict" in fit_res.metrics:
                p_k_t_state_dict_bytes = fit_res.metrics.pop("p_k_t_state_dict")
                p_k_t_state_dict_numpy = pickle.loads(p_k_t_state_dict_bytes)

                old_hpm_state = self.client_hpms.get(logical_cid)
                if old_hpm_state is None:
                    self.client_hpms[logical_cid] = p_k_t_state_dict_numpy
                else:
                    self.client_hpms[logical_cid] = update_hpm_state_dict(
                        old_hpm_state, p_k_t_state_dict_numpy, self.hpm_beta
                    )

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self, server_round: int, results: List[Tuple[ClientProxy, fl.common.EvaluateRes]], failures: List[Union[Tuple[ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results."""
        if not results:
            return None, {}

        print(f"\n[Round {server_round}] Detailed evaluation results:")
        for client_proxy, res in results:
            logical_cid = res.metrics.get("logical_id", client_proxy.cid)
            acc_personalized = res.metrics.get("accuracy_personalized", "N/A")
            acc_hpm = res.metrics.get("accuracy_hpm", "N/A")
            print(
                f"  - Client {logical_cid}: "
                f"Personalized Acc = {acc_personalized:.4f}, "
                f"HPM Acc = {acc_hpm:.4f}"
            )

        num_total_examples = sum(res.num_examples for _, res in results)
        loss_aggregated = sum(res.loss * res.num_examples for _, res in results) / num_total_examples
        accuracy_personalized_aggregated = sum(res.metrics["accuracy_personalized"] * res.num_examples for _, res in results) / num_total_examples
        accuracy_hpm_aggregated = sum(res.metrics["accuracy_hpm"] * res.num_examples for _, res in results) / num_total_examples
        metrics_aggregated = {
            "accuracy_personalized": accuracy_personalized_aggregated,
            "accuracy_hpm": accuracy_hpm_aggregated,
        }
        return loss_aggregated, metrics_aggregated
