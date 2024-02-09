import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, r2_score
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
import heapq
from util.helpers import playlistToSparseMatrixEntry, getPlaylistTracks
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobPrefix
import io


class NNeighClassifier():
    def __init__(self, playlists, sparsePlaylists, songs, reTrain=False, name="NNClassifier.pkl", blob_service_client=None):
        self.pathName = name
        self.name = "NNC"
        self.playlistData = sparsePlaylists
        self.playlists = playlists
        self.songs = songs
        self.credential = DefaultAzureCredential()
        self.blob_service_client = blob_service_client
        self.initModel(reTrain)
    
        
    def initModel(self, reTrain):
        """
        """
        blob_client = self.blob_service_client.get_blob_client("data", f"lib/{self.pathName}")
        
        exists = blob_client.exists()
        print(exists)
        print(reTrain)
        if not exists or reTrain:
            print("Creating new model")
            self.model = NearestNeighbors(
                n_neighbors=60,
                metric="cosine")
            self.trainModel(self.playlistData)
        else:
            print("Downloading model from blob storage")
            blob_data = blob_client.download_blob().readall()
            self.model = pickle.loads(blob_data)
        

    def trainModel(self, data):
        """
        """
        print(f"Training Nearest Neighbors classifier")
        self.model.fit(data)
        self.saveModel()
        

    def getNeighbors(self, X, k):
        """
        """
        return self.model.kneighbors(X=X, return_distance=False, n_neighbors=k)[0]

    def getPlaylistsFromNeighbors(self, neighbours, pid):
        """
        """
        neighbours = list(filter(lambda x: x != pid, neighbours))
        return [self.playlists.loc[x] for x in neighbours]

    def getPredictionsFromTracks(self, tracks, numPredictions, pTracks):
        """
        """
        pTracks = set(pTracks)
        songs = defaultdict(int)
        for i, playlist in enumerate(tracks):
            for song in playlist:
                track_uri = song['track_uri'].split(":")[2]
                if track_uri not in pTracks:
                    songs[track_uri] += (1/(i+1))
        scores = heapq.nlargest(numPredictions, songs, key=songs.get)
        return scores
        # return list(predictedSet)

    def predict(self, X, numPredictions, songs, numNeighbours=60):
        """
        """
        pid, pTracks = X["pid"], X["tracks"]
        sparseX = playlistToSparseMatrixEntry(X, self.songs)
        neighbors = self.getNeighbors(sparseX, numNeighbours)  # PlaylistIDs
        playlists = self.getPlaylistsFromNeighbors(neighbors, pid)
        tracks = [getPlaylistTracks(x, self.songs) for x in playlists]
        predictions = self.getPredictionsFromTracks(
            tracks, numPredictions, pTracks)
        return predictions

    def saveModel(self):
        """
        """        
        print("pickle the model")

        model_pickle = io.BytesIO()
        pickle.dump(self.model, model_pickle)
        model_pickle.seek(0)
        print("uploading model to blob storage")
        self.blob_service_client.get_container_client(container = "data").upload_blob(name = "lib/NNClassifier.pkl", data = model_pickle, overwrite=True)
        print("model pickled")
        