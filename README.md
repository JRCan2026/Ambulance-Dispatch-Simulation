# Ambulance-Dispatch-Simulation
<img width="1536" height="1024" alt="ambulance_IA" src="https://github.com/user-attachments/assets/b69f2bdc-50ef-44c8-9d06-79547f37ef2b" />
In this project two ambulance dispatch strategies are compared to determine which of them performs best in reducing response and waiting times when serving incoming emergency calls. The policies considered are:

1) Greedy Policy (GP): assigns the closest available ambulance to each incoming emergency call, minimizing immediate travel time.
2) Optimized assignation policy (OP): assigns ambulances by solving a global allocation problem at each timestep, aiming to minimize total travel time across assignments in the system.

**Simulation setup**
* Initial ambulance positions are randomly assigned within the boundaries of Québec City and Lévis.
* Six hospitals are selected randomly from a set of nine actual facilities in Québec City and Lévis.
* Emergency calls are generating using a Poisson process with $\lambda=30\ calls/hour$.
* The simulation runs for 60 minutes with a 1-minute time step.

Instead of relying on summary statistics such as means or percentiles, survival analysis is used on response and waiting time distributions tp compare policy performance. This enables identification of efficiency regimes, capturing not only which patching strategy performs better on average, but also where and in what proportion differences arise across the distribution.

Finally, a sensitivity analysis is conducted by varying the fleet size to assess system response under different levels of operational stress.

