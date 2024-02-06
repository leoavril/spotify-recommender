from flask import Flask, request, render_template, redirect, url_for
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from main import SpotifyExplorer  # import the SpotifyExplorer class from your main.py file
import os
import json
import pandas as pd

app = Flask(__name__)

# Initialize the pid
pid = 1000000

# Print all the tracks that have been sugested
tracks_embed = []

spotify_explorer = SpotifyExplorer(numFiles=0, retrainNNC=False)  # initialize the SpotifyExplorer

# Set up Spotipy
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="9858097f4b8c4a3dbec46bbc0cf04186",
                                                           client_secret="d6c0234cad074c45bc3baf0e56c6b4bb"))

@app.route('/')
def index():
    tracks_embed = request.args.getlist('tracks_embed')  # Get list from query parameters
    print(tracks_embed)
    return render_template('index.html', tracks_embed=tracks_embed)

@app.route('/predict', methods=['POST'])
def predict():
    global pid  # declare pid as global so we can modify it
    global tracks_embed

    # Get the playlist_id from the request data
    playlist_id = request.form.get('playlist')

    # Initialize SpotifyExplorer
    spotify_explorer = SpotifyExplorer(numFiles=0, retrainNNC=True)

    # Get playlist details
    playlist_details = sp.playlist(playlist_id)

    # Get tracks
    raw_tracks = playlist_details['tracks']['items']

    # Transform the tracks into the desired format
    tracks = [{
        'pos': i,
        'artist_name': track['track']['artists'][0]['name'],
        'track_uri': track['track']['uri'],
        'artist_uri': track['track']['artists'][0]['uri'],
        'track_name': track['track']['name'],
        'album_uri': track['track']['album']['uri'],
        'duration_ms': track['track']['duration_ms'],
        'album_name': track['track']['album']['name']
    } for i, track in enumerate(raw_tracks)]

    
    # Create a suitable JSON dictionary
    playlist = {
        'info': {
            'generated_on': '2017-12-04 03:05:11.774401', 
            'slice': f'{pid}-{pid+999}', 
            'version': 'v1'
        },
        'playlists': [{
            'name': playlist_details['name'],
            'collaborative': playlist_details['collaborative'],
            'pid': pid,
            'modified_at': 1468800000,  
            'num_tracks': len(tracks),
            'num_albums': len(set([track['album_uri'] for track in tracks])),
            'num_followers': playlist_details['followers']['total'],
            'tracks': tracks,
            'num_edits': 2,  
            'duration_ms': sum([track['duration_ms'] for track in tracks]),
            'num_artists': len(set([track['artist_uri'] for track in tracks]))
        }]
    }
    
    # create pandas Series object


    playlist_series = pd.Series({
        'name': playlist_details['name'],
        'collaborative': playlist_details['collaborative'],
        'pid': pid,
        'modified_at': 1468800000,
        'num_tracks': len(tracks),
        'num_albums': len(set(track['album_uri'] for track in tracks)),
        'num_followers': playlist_details['followers']['total'],
        'tracks': [track['track_uri'].split(':')[-1] for track in tracks],
        'num_edits': 2,
        'duration_ms': sum(track['duration_ms'] for track in tracks),
        'num_artists': len(set(track['artist_uri'] for track in tracks)),
        'description': None  # replace with actual description if available
    }, name=pid)

    # Create the data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    # Write the playlist to a new JSON file
    with open(f'data/mpd.slice.{pid}-{pid+999}.json', 'w') as f:
        json.dump(playlist, f, indent=4)

    # Increment the pid for the next playlist
    pid += 1000

    # Make a prediction and get the embed links of the suggested tracks
    
    print(playlist_series)
    tracks_embed  = spotify_explorer.PredictPlaylist(playlist_series)
    # Redirect to the index page and pass the embed_link to the template   
    return redirect(url_for('index', tracks_embed=tracks_embed))

    

if __name__ == '__main__':
    app.run(debug=True)
    