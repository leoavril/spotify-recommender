import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())

SPOTIFY_CLIENT = '9858097f4b8c4a3dbec46bbc0cf04186'
SPOTIFY_CLIENT_SECRET = 'd6c0234cad074c45bc3baf0e56c6b4bb'
SPOTIFY_REDIRECT_URI = 'http://localhost:5000/callback'

def createRandomPredictionsDF(self, numInstances):
        print(f"Generating {numInstances} data points")
        data = [self.displayRandomPrediction() for _ in tqdm(range(numInstances))]
        df = pd.DataFrame(data)
        df.to_csv("predictionData.csv")

    # Display prediction for given playlist and return as dictionary

    def displayRandomPrediction(self):
        playlist = self.getRandomPlaylist()
        while len(playlist["tracks"]) < 10:
            playlist = self.getRandomPlaylist()

        predictions = self.predictNeighbour(playlist=playlist,
            numPredictions=50,
            songs=self.songs)

        playlistName = playlist["name"]
        playlist = [getTrackandArtist(trackURI, self.songs) for trackURI in playlist["tracks"]]
        predictions = [getTrackandArtist(trackURI, self.songs) for trackURI in predictions]
        return {
            "name": playlistName,
            "playlist": playlist,
            "predictions": predictions
        }
    
"""
    def evalAccuracy(self, numPlaylists, percentToObscure=0.5): 
        print()
        print(f"Selecting {numPlaylists} playlists to test and obscuring {int(percentToObscure * 100)}% of songs")

        def getAcc(pToObscure):
            playlist = self.getRandomPlaylist()

            keptTracks, obscured = obscurePlaylist(playlist, pToObscure)
            playlistSub = playlist.copy()
            obscured = set(obscured)
            playlistSub['tracks'] = keptTracks

            predictions = self.predictNeighbour(playlistSub, 
                500, 
                self.songs)

            overlap = [value for value in predictions if value in obscured]

            return len(overlap)/len(obscured)
        
        accuracies = [getAcc(percentToObscure) for _ in tqdm(range(numPlaylists))]
        avgAcc = round(sum(accuracies) / len(accuracies), 4) * 100
        print(f"Using {self.classifier.name}, we predicted {avgAcc}% of obscured songs")"""
    
