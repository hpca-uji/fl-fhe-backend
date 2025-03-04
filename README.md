# CIBER-CAFE

Research project entitled "CIBERseguridad post-Cuántica para el Aprendizaje FEderado en procesadores de bajo consumo y aceleradores" (Post-Quantum Cybersecurity for FEderated Learning on Low-Power Processors and Accelerators).

**Project website:** https://sites.google.com/uji.es/ciber-cafe/home

## Project data

**ID:** C121/23

**FUNDING ENTITY:** Instituto Nacional de Ciberseguridad de España

**PROGRAMME:** Proyectos Estratégicos de Ciberseguridad en España 2022

**PERIOD:** 01/01/2024 - 31/12/2025

**NUMBER OF INVESTIGATORS:** 6

## Contact information

This project belongs to the HPC&A research group from the Universitat Jaume I (Spain).

For more information regarding the project's contribution, please contact:

- Manuel F. Dolz Zaragozá (dolzm@uji.es)

- Sandra Catalán Pallarés (catalans@uji.es)

- Darwin Quezada (quezada@uji.es)
  
### Run server

#### Run the Superlink
This command starts the **Superlink**, which acts as the central communication hub for the federated learning system.

````
flower-superlink --insecure
````

#### Run the server App
This starts the server application for federated learning.

````
flwr run . local-deployment --stream
````

#### Run the super node
The supernode facilitates communication between clients and the federated learning server. If it is running locally `--clientappio-api-address 127.0.0.1:9095` and change the port, otherwise remove the line.
````
flower-supernode \
     --insecure \
     --superlink 127.0.0.1:9092 \
     --clientappio-api-address 127.0.0.1:9095 \ 
     --node-config "partition-id=1 num-partitions=2"
````

#### Run the client
To add more clients to the **SuperNode**
```
flwr-clientapp --clientappio-api-address 127.0.0.1:<supernode-port> --insecure
```