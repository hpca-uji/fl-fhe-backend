from typing import Dict, Callable

def on_fit_config(epoch=2, lr=0.001, batch_size=32) -> Callable[[int], Dict[str, str]]:
    def fit_config(server_round: int) -> Dict[str, str]:
        config = {
            "learning_rate": str(lr),
            "batch_size": str(batch_size),
            "server_round": server_round,
            "local_epochs": epoch
        }
        return config
    return fit_config