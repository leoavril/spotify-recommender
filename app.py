from flask import Flask, request, abort, render_template, redirect, url_for, Response
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
# import the SpotifyExplorer class from your main.py file
from main import SpotifyExplorer
import os
import json
import pandas as pd
from util.helpers import parsePlaylistLink

app = Flask(__name__)

# Initialize the pid
pid = 1000000

# Print all the tracks that have been sugested
tracks_embed = []

# initialize the SpotifyExplorer
spotify_explorer = SpotifyExplorer(numFiles=0, retrainNNC=False)

# Set up Spotipy
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="f2458f8ee1304f43acbb7f6037de86c8",
                                                           client_secret="cd9f1486ebad4d088766160b532d04de"))


@app.errorhandler(400)
def not_playlist_link(e):
    return render_template('index.html'), 400


@app.route('/')
def index():
    return render_template('index.html', tracks_embed=tracks_embed)


@app.route('/predict', methods=['POST'])
def predict():
    global pid  # declare pid as global so we can modify it
    global tracks_embed

    # Get the playlist_id from the request data
    playlist_link = request.form.get('playlist')
    print(playlist_link)
    # Initialize SpotifyExplorer
    # spotify_explorer = SpotifyExplorer(numFiles=0, retrainNNC=True)

    try:
        parsedPlaylist = parsePlaylistLink(sp, playlist_link, pid)
        playlist = parsedPlaylist[0]
        playlist_series = parsedPlaylist[1]
    except:
        abort(400)

    # Create the data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    # Write the playlist to a new JSON file
    with open(f'data/mpd.slice.{pid}-{pid+999}.json', 'w') as f:
        json.dump(playlist, f, indent=4)

    # Increment the pid for the next playlist
    pid += 1000

    # Make a prediction and get the embed links of the suggested tracks

    tracks_embed = spotify_explorer.predictPlaylist(playlist_series)
    # Redirect to the index page and pass the embed_link to the template
    return redirect(url_for('index', tracks_embed=tracks_embed))


if __name__ == '__main__':
    app.run(debug=True)
