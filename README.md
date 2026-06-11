# Ambulance-Dispatch-Simulation
<img width="1536" height="1024" alt="ambulance_IA" src="https://github.com/user-attachments/assets/b69f2bdc-50ef-44c8-9d06-79547f37ef2b" />
In this project two ambulance dispatch strategies are compared to determine which of them proves best at reducing response and waiting times when serving incoming emergency calls. The policies compared are listed below:
1) **Greedy Policy (GP)**: assigns the closest available ambulance to each incoming emergency call, minimizing immediate travel time.
2) **Optimized assignation policy (OP)**: assigns ambulances by solving a global allocation problem at each timestep, aiming to minimize total travel time across assignments in the system.

**Simulation setup**
* Initial ambulance positions are assign randomly within the boundaries of Quebec City and Levis
* Six hospital are choosen randomly within a set of 9 actuall facilities in Quebec and Levis
* Emergency calls are generating using a Poisson process with $\lambda=30\ call/h$
* The simulation last 60 minutes with timestep of 1 minute

Instead of summary statistics such as mean or percentiles, survival analysis is used over the response time and waiting time distributions in the comparison of policies' performance, allowing likewise the identification of efficiency regimes, to know not only which patching strategy is better in average but where and in which proportion respect to the other.

Finally, sensitivity analysis is conducted varying the fleet size to test system's response under different levels of stress.

