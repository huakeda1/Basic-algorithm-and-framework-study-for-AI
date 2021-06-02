#!/usr/bin/env python
# coding: utf-8

# The following three problems will be solved by the same graph searching method(breadth search and depth search)
# 
# 1) Iterate through all restaurants and find the shortest path between the start point and end point.  
# 2) Pick up and pour water back and forth so as to get specific volume by two cups of known volume.  
# 3) Search through the subway station map obtained by crawler and find the shortest path or the least transfering path between the start station and the final station.

# In[1]:


import pickle
import requests
from collections import defaultdict
from bs4 import BeautifulSoup


# In[2]:


restaurant_dict={
    "全聚德":"重庆小面 湖南米粉 龙抄手".split(),
    "重庆小面":"龙抄手 便宜坊".split(),
    "便宜坊":"牛排 重庆小面".split(),
    "港式火锅": "云水谣 龙抄手".split(),
    "海底捞":"港式火锅".split(),
}


# In[3]:


from collections import defaultdict
def get_mapping_dict(raw_map):
    new_map=defaultdict(set)
    for node,children in raw_map.items():
        new_map[node]=set(children)
        for element in children:
            new_map[element].add(node)
    return {node:list(sets) for node,sets in new_map.items()}


# In[4]:


def goal_found(node):
    def _wrap(path):
        return node==path[-1] or node in path[-1]
    return _wrap


# In[5]:


def node_mapping(map_dict):
    def _wrap(node):
        return {item:'=>' for item in map_dict[node]}
    return _wrap


# In[6]:


def state_mapping(A,B):
    def _wrap(state):
        a,b=state
        return {(0,b):'清空A',
                (a,0):'清空B',
                (A,b):'倒满A',
                (a,B):'倒满B',
                (0,a+b) if a+b<B else (a+b-B,B):'a=>b',
                (a+b,0) if a+b<A else (A,a+b-A):'b=>a'}
    return _wrap


# In[7]:


def station_mapping(station_dict):
    def _wrap(node):
        later_node_mapping={}
        for line,stations in station_dict.items():
            if node in stations:
                index=stations.index(node)
                if index>=1:
                    later_node_mapping[stations[index-1]]='{}'.format(line)+'=>'
                if index+1<len(stations):
                    later_node_mapping[stations[index+1]]='{}'.format(line)+'=>'
        return later_node_mapping
    return _wrap


# In[8]:


def search(start,goal_func,mapping_func,optim_func):
    paths=[[start]]
    explored=set()
    while paths:
        # 0 here means breadth search; -1 means depth search
        path=paths.pop(0)
        last_state=path[-1]
        if last_state in explored:continue
        for state,action in mapping_func(last_state).items():
            if state in explored:continue
            new_path=path+[action,state]
            if goal_func(new_path):
                return new_path
            paths.append(new_path)
        explored.add(last_state)
        paths=sorted(paths,key=optim_func)
    return []

def iterate(start,mapping_func):
    path=[start]
    visited=[]
    explored=set()
    while path:
        # 0 here means breadth search; -1 means depth search
        last_state=path.pop(0)
        if last_state in explored:continue
        visited.append(last_state)
        for state,action in mapping_func(last_state).items():
            if state in explored:continue
            path.append(state)
        explored.add(last_state)
    return visited


# In[9]:


all_visited_nodes=iterate(start='全聚德',mapping_func=node_mapping(get_mapping_dict(restaurant_dict)))
print('all visited nodes:',all_visited_nodes)


# In[10]:


search(start='全聚德',goal_func=goal_found('港式火锅'),
       mapping_func=node_mapping(get_mapping_dict(restaurant_dict)),
       optim_func=lambda x:len(x))


# In[11]:


search(start=(0,0),goal_func=goal_found(60),
       mapping_func=state_mapping(90,40),
       optim_func=lambda x:len(x))


# In[12]:


# here is used to count time
# from tqdm import tqdm
# import time
# for t in tqdm(range(60*1)):
#     time.sleep(1)


# In[13]:


def get_subway_station_map(url):
    response=requests.get(url)
    response.encoding='gbk'
    soup=BeautifulSoup(response.text,'html.parser')
    lines=soup.find_all(attrs={'class':'line_name'})
    station_dict=defaultdict(list)
    for line in lines:
        line_name=line.text.strip()
        for s in line.next_siblings:
            if s.string and BeautifulSoup(s.string).find_all(attrs={'class':'line_name'}):break
            text=s.string
            if text and text.strip():
                station=text.strip()
                station_dict[line_name].append(station)
    return dict(station_dict)

def get_subway_station_map(url):
    response=requests.get(url)
    response.encoding='gbk'
    soup=BeautifulSoup(response.text,'html.parser')
    #lines=soup.find_all(attrs={'class':'line_name'})
    line_stations=soup.find_all('div',attrs=['line_name','station'])
    # print(list(line_stations))
    station_dict=defaultdict(list)
    for line_station in line_stations:
        if 'line_name' in str(line_station):
            line=line_station.text.strip()
        if 'station' in str(line_station):
            station_dict[line].append(line_station.text.strip())
    return dict(station_dict)


# In[14]:


station_dict=get_subway_station_map(url='https://www.bjsubway.com/station/xltcx/')
station_dict


# In[15]:


def subway_path_parse(path):
    output_text=[]
    start=path[0]
    line=path[1]
    for index,item in enumerate(path):
        if index%2!=0 and line!=item:
            end=path[index-1]
            action='上乘' if not output_text else '换乘'
            current_path='{}{}:从{}坐到{}'.format(action,line[:-2],start,end)
            output_text.append(current_path)
            start=path[index-1]
            line=item
    action='上乘' if not output_text else '换乘'
    current_path='{}{}:从{}坐到{}'.format(action,line[:-2],start,path[-1])
    output_text.append(current_path)
    return '=>'.join(output_text)


# In[16]:


# least stations to get the destination
result1=search(start='苹果园',goal_func=goal_found('苏庄'),
       mapping_func=station_mapping(station_dict),
       optim_func=lambda x:len(x))
print('detailed info is shown as below:')
print(result1)
print('simple info is shown as below:')
print(subway_path_parse(result1))

# least line to get the destination
result2=search(start='苹果园',goal_func=goal_found('苏庄'),
       mapping_func=station_mapping(station_dict),
       optim_func=lambda x:len(set([item for index,item in enumerate(x) if index%2!=0])))
print('detailed info is shown as below:')
print(result2)
print('simple info is shown as below:')
print(subway_path_parse(result2))


# In[ ]:




