from pymongo import MongoClient
from models.agent import Agent

def get():
    a = Agent()
    return a

if __name__ == '__main__':
    b = get()
    print(b)

