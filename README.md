# Spaceship-Landing

## train network
### qlearning + greedy epsilon

python train.py --agents qlearning --number_episodes 3000 --lr 0.0001 --gamma 0.99 --final_eps 0.02 --max_steps_per_episodes 5000

### qlearning + greedy epsilon

python train.py --agents qlearning_bolt --number_episodes 3000 --lr 0.0001 --gamma 0.99 --final_eps 0.02 --max_steps_per_episodes 5000

### Double Deep Q-learning + epsilon-greedy

python train.py --agents deepqn --number_episodes 3000 --lr 0.0001 --gamma 0.99 --final_eps 0.02 --max_steps_per_episodes 5000

### Double Deep Q-learning + boltzmann exploration

python train.py --agents dqn_bolt --number_episodes 3000 --lr 0.0001 --gamma 0.99 --final_eps 0.02 --max_steps_per_episodes 5000
