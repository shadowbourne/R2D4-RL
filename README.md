# Recurrent Replay Deeper Denser Distributed DQN+ (Reinforcement Learning):

This work was undertaken and submitted as a 3rd year Bachelors Dissertation/Project as part of the degree of Msci Natural Sciences to the Board of Examiners in the Department of Computer Sciences, Durham University, and is licensed accordingly. 
## Grade: 1st - 92/100

This agent was trained and extensively tested on Super Mario Bros NES, credits for the enviroment go to [Christian Kauten](https://github.com/Kautenja/gym-super-mario-bros). See main paper and codebase for all references and accreditatons.

## Repository Contents:
* The main 20 page paper, Combining Recent Advances in Reinforcement Learning for Super Mario Bros.;
* A 3 page Appendix of additional theory and explanations;
* The Project Demonstration (a 30 minute Oral Exam demonstrating the work undertaken and success of the Project);
* The accompanying codebase for the agent R2D4+, with switches to turn on and off every individual component (very useful for learning what individual papers have contributed, if you are new to the field);
* An extensive base of training videos taken throughout training of the agent on various training sets and agent component set-ups (submit an issue if you have any queries regarding run names), very useful to see how the agent learns, what it struggles with ;
* Interactive graphs for all data gathered from (most) training runs ([click here](https://tensorboard.dev/experiment/EyHmIrtQSyarXQGXzVL4gw/ "TensorBoard Graphs")), hosted on TensorBoard.dev, alongside most of the raw TensorBoard data too to be used for future studies and learning. Once again, get in contact if you have any queries regarding run names :).

## Demo (taken from Final Presentation):
  > ![Gifdemo6](https://user-images.githubusercontent.com/18665030/136659042-7b071648-e77a-4fef-9757-9160d22ae2b0.gif)
  > 
  > During my Dissertation, I surveyed the field of Reinforcement Learning and spent my project investigating how best to combine many recent advances from Reinforcement Learning. In doing so, I created an AI agent capable of teaching itself to play the game Super Mario Bros. to superhuman level, achieving new state-of-the-art performance within the field. This demo shows the agent playing Worlds 5 and 7, and was taken from the project's [final presentation](https://github.com/shadowbourne/R2D4-RL/blob/main/Demonstration%20Final%20Presentation.pptx).
  > 
  > <img src="https://user-images.githubusercontent.com/18665030/136562342-ef434a45-5ffd-4d7c-858e-ceee3894eec4.png" width="48" title="Mario"/>

## Abstract:
### Context:
A  variety  of,  predominantly  Reinforcement  Learning  (RL)  based,  approaches have been applied over the years to creating an intelligent agent able to play the platformer game Super Mario Bros (SMB), however,  the majority are heavily reliant upon expert feature-engineering and low-dimensional state-encodings to make learning tractable without the use of Neural Networks.  This application has been largely untouched by any recent state-of-the-art RL contributions.
### Aims:
To investigate how best to apply and combine recent advances in deep RL so as to create a highly proficient agent capable of teaching itself to play SMB using raw-image-data without any feature-engineering.
### Method:
After surveying the field, a Recurrent Replay Deeper Denser Distributed DQN+ (R2D4+) (as named and defined within this paper) agent has been created by attempting to combine the constituent components of recent RL architectures, namely R2D2, Rainbow, D2RL and IQN, alongside some novel contributions such as dropout and an improved exploration scheme.
### Results:
The final agent is highly proficient, surpassing all previously surveyed agents, and converges to or surpasses the skill and speed of an expert on a variety of level difficulties and training-set sizes. The agent also generalizes impressively well to levels it has never been trained on.
### Conclusions:
All surveyed contributions, apart from PER and IQN, are shown to be complementary in the context of training a RL agent to play SMB, which provides evidence that further investigation and combination of additional contributions would prove beneficial to the development of the field of RL. The novel use of dropout to boost generalization power and sample-efficiency is especially promising. Overall, deep RL in its current state can produce excellent self-teaching intelligent agents and is still a field of much promise for the future.

