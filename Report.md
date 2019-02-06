[image2]: score.png "Score"

### Learning Algorithm

This project uses the DDPG training algorithm which is considered an actor-critic algorithm. It was first introduced by [this paper](https://arxiv.org/abs/1509.02971). This method is a development over the Deep Q Learning method so that it can be used in a continuous action domain. The paper states that it is impossible to apply Q Learning to continuous action spaces because finding the greedy policy requires an optimization of a<sub>t</sub> at every time step and this optimization is too slow thus rendering this method unusable. Instead, they follow an actor-critic method
based on DPG.

The DPG algorithm maintains an actor function which specifies the current policy deterministically mapping states to specific actions. The critic Q(s,a) is learned using the Bellman's equation. Since using Neural Networks as nonlinear function approximators makes the convergence guarantees dissapear, techniques such as Minibatch training and Replay Buffers are used. Check the paper for more details.

The model used for the actor is the following :
```python
self.fc1 = nn.Linear(state_size, fc1_units)
self.fc2 = nn.Linear(fc1_units, fc2_units)
self.fc3 = nn.Linear(fc2_units, action_size)
```
and the forward pass for the actor:

```python
x = F.relu(self.fc1(state))
x = F.relu(self.fc2(x))
return F.tanh(self.fc3(x))
```

We are using the tanh activation function to model the continuous output for the agent's action.

The model used for the critic is the following:
```python
self.fcs1 = nn.Linear(state_size, fcs1_units)
self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
self.fc3 = nn.Linear(fc2_units, 1)
```
with the following forward pass:
```python
xs = F.relu(self.fcs1(state))
x = torch.cat((xs, action), dim=1)
x = F.relu(self.fc2(x))
return self.fc3(x)
```
We use 400 neurons for the first hidden layer and 300 neurons for the second hidden layer for both the actor and the critic. The full code is in [model.py](model.py) 

This project uses the 20 agent environment. The full code for the agent can be found in [ddpq_agent_20.py](ddpq_agent_20.py). The agent takes the state, action , reward ,next_state from all 20 different agents and if the game finished and saves them for future learning. Once enough experiences have accumulated, the agent randomly samples from then and trains the neural network described above. This act of sampling is called Experience Replay. We could've trained the neural network using all the previous steps , but the sequence of exerience tupples is highly correlated and we risk getting swayed by the effects of this correlation. By randomly selecting episodes, we break the correlation. I've used a buffer size of 100k in order to store a large enough number of steps for the agent to remember properly, and a batch size of 1024 for every training step.

The problem is considered solved when the average score is 30 over the last 100 episodes. This algorithm is able to train the agent pretty quickly (298 episodes in my case).
The following graph represents the score of the agent over episodes :

![Score][image2]

The algorithm could be further improved by ,for example, changing the sampling of the steps used for learning. Currenly we sample uniformly from the Replay Buffer, but we could do prioritized experince replay, which means that we give higher weight more meaningful transitions, thus sampling them more frequently.

If you'd like to see how the solved environment works, load up [TrainedModel.ipynb]([TrainedModel.ipynb]) and give it a go.
