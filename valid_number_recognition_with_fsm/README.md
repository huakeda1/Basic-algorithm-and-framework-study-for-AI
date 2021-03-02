## Effective number recognition of string with FSM
You are going to determine whether the string is a valid number, this is actually a typical application of the principle of FSM(finite state machine), you should define all possible states and their transition paths，one type of char and one state can get you through next definite state, you will analyze the string char by char util you get the final state, if the final state is in all acceptable states, then the string is considered to be a valid number.  
Examples for valid number:  
["2", "0089", "-0.1", "+3.14", "4.", "-.9", "2e10", "-90E3", "3e+7", "+6e-1", "53.5e93", "-123.456e789"]  
Examples for invalid number:  
["abc", "1a", "1e", "e3", "99e2.5", "--6", "-+3", "95a54e53"]  
Scheduled transfering paths between different states:  
![transfering_paths](https://raw.github.com/huakeda1/Basic-algorithm-and-framework-study-for-AI/master/valid_number_recognition_with_fsm/associated_pngs/transfering_paths.png)  
This task is from [LeetCode](https://leetcode-cn.com/problems/valid-number),you can get more solution from this link.