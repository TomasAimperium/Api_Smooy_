from pymongo import MongoClient

HOST = "mongodb+srv://tomasalcaniz:AcYT31HxtJ1FMKKt@aicluster.2d6o5fg.mongodb.net/?retryWrites=true&w=majority"

client = MongoClient(host=HOST)

client.admin.command('ping')

db = client.tomasalcaniz