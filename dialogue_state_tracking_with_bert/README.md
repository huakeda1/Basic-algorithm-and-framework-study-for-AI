# Dialogue state tracking based on Bert
There are many turns in one dialogue, **'transcript'** and **'system_transcript'** info in one turn will used to get request or inform states, the predicted states will be compared with **'turn_label'** info in one turn to get the **'turn_inform'** score or **'turn_request'** score, these states called tracking states will be added or updated as turn goes on, they will be compared with **'belief_state'** so as to get **'joint_goal'** score.  
The state recognition problem will be treated as **binary classification** case，One turn will be transformed to many examples, the label is True or False in each example,the loss function is **torch.nn.BCELoss()**, all examples of turns will be collected together for training batch by batch, all examples in one turn will be carried out binary classification and the label with higher probability will be remained to get the final result of one turn in prediction process, all results of turns will be later post processed to get relevant evaluating score.  

## Model Structure
The figure below shows the architecture of the bert based model:
![model_architecture](https://raw.github.com/huakeda1/Basic-algorithm-and-framework-study-for-AI/master/dialogue_state_tracking_with_bert/associated_pngs/simple_bert_model_for_dst.png)  

## Running Commands
An example training command is:  
`python main.py --do_train --data_dir=data/woz/ --bert_model=bert-base-uncased --output_dir=outputs`  
An example evaluating command is:  
`python main.py --do_eval --data_dir=data/woz/ --bert_model=bert-base-uncased --output_dir=outputs`  

## Evaluating Results
Number of Train Dialogues: 600;Number of Dev Dialogues: 200;Number of Test Dialogues: 400
As for dev dialogues:  
{'turn_inform': 0.944578313253012, 'turn_request': 0.9746987951807229, 'joint_goal': 0.9036144578313253}
As for test dialogues:  
{'turn_inform': 0.9362089914945322, 'turn_request': 0.9817739975698664, 'joint_goal': 0.9003645200486027}

## Code Structure
The figure below shows the architecture of the whole code:
![code_architecture](https://raw.github.com/huakeda1/Basic-algorithm-and-framework-study-for-AI/master/dialogue_state_tracking_with_bert/associated_pngs/Dialogue_state_tracking_with_bert.png)  
You can get more detailed information from this [link](https://naotu.baidu.com/file/17947dc4dfba2a8d824b06981d0170f8)