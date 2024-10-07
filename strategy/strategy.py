import logging
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import EvaluateRes, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

from prometheus_client import Gauge

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module


class FedCustom(fl.server.strategy.FedAvg):
    def __init__(
        self, cos_gauge: Gauge = None, mae_gauge: Gauge = None, loss_gauge: Gauge = None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        #self.accuracy_gauge = accuracy_gauge
        self.cos_gauge = cos_gauge
        self.mae_gauge = mae_gauge
        self.loss_gauge = loss_gauge

    def __repr__(self) -> str:
        return "FedCustom"

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses and accuracy using weighted average."""

        if not results:
            return None, {}

        # Calculate weighted average for loss using the provided function
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Calculate weighted average for accuracy
        coss = [
            evaluate_res.metrics["cos"] * evaluate_res.num_examples
            for _, evaluate_res in results
        ]
        maes = [
            evaluate_res.metrics["mae"] * evaluate_res.num_examples
            for _, evaluate_res in results
        ]

        examples = [evaluate_res.num_examples for _, evaluate_res in results]
        # all the metrics aggregated
        cos_aggregated = (
            sum(coss) / sum(examples) if sum(examples) != 0 else 0
        )
        mae_aggregated = (
            sum(maes) / sum(examples) if sum(examples) != 0 else 0
        )

        # Update the Prometheus gauges with the latest aggregated accuracy and loss values
        self.cos_gauge.set(cos_aggregated)
        self.mae_gauge.set(mae_aggregated)
        self.loss_gauge.set(loss_aggregated)

        metrics_aggregated = {"loss": loss_aggregated, "cos": cos_aggregated, "mae": mae_aggregated}

        return loss_aggregated, metrics_aggregated
