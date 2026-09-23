import tomli

class LoadConfig:
  def __init__(self, config_file) -> None:
    with open(config_file, "rb") as f:
      self.config = tomli.load(f)
      
  def get_app_config(self):
    root = self.config["tool"]["flwr"]["app"]["config"]
    return {
      "client" : root["client"],
    }
    
  def get_he_config(self):
    root = self.config["tool"]["federated_learning"]["he"]
    return {
      "enable" : root["enable"],
      "context" : {
        "client": root["context"]["path"]["client"],
        "server": root["context"]["path"]["server"],
      },
      "key" : {
        "client" : root["key"]["path"]["client"],
        "server" : root["key"]["path"]["server"], 
        "server_crypted" : root["key"]["path"]["server-crypted"]
      },
      "library": root["library"],
      "schema" : root["schema"],
      "poly_modulus_degree" : root["poly-modulus-degree"],
      "coef_mod_bit" : root["coeff-mod-bit-sizes"],
      "global_scale" : root["global-scale"]
    }
  
  def get_nn_config(self):
    root = self.config["tool"]["federated_learning"]["nn"]
    return {
      "model" : root["model"],
      "store_model" : root["store-model"],
      "seed" : root["seed"],
      "max_epochs" : root["max-epochs"],
      "batch_size" : root["batch-size"],
      "learning_rate" : root["learning-rate"],
      "model_path" : root["storage-model-path"],
      "model_save" : root["model-save"],
      "checkpoint" : root["checkpoint-save"]
    }
    
  def get_test_config(self):
    root = self.config["tool"]["federated_learning"]["test"]["dataset"]
    return {
      "dataset" : root["dataset"],
      "test_dataset" : root["test-dataset"],
      "splitter" : root["splitter"],
      "num_clients" : root["num-clients"],
      "num_workers" : root["num-workers"],
      "batch_size" : root["batch-size"],
      "resize" : root["resize"],
      "seed" : root["seed"],
      "data_path" : root["data-path"],
      "validation_data" : root["validation-data"],
      "classes" : root["classes"],
      "normalization" : {
        "mean" : root["normalization"]["mean"],
        "std" : root["normalization"]["std"]
      }
    }
    
  def get_stats_config(self):
    root = self.config["tool"]["federated_learning"]["stats"]
    return {
      "enable" : root["enable"],
      "plots" : root["path"]["plots"],
      "reports" : root["path"]["reports"]
    }
    
  def get_quantum_config(self):
    root = self.config["tool"]["federated_learning"]["quantum"]
    return {
      "enable" : root["enable"],
      "num_qubits" : root["num-qubits"],
      "num_layers" : root["num-layers"]
    }