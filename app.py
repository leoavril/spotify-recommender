from flask import Flask, request, abort, render_template, redirect, url_for, Response
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
# import the SpotifyExplorer class from your main.py file
from main import SpotifyExplorer
import os
import json
import pandas as pd
from util.helpers import parsePlaylistLink
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient


app = Flask(__name__)

# Print all the tracks that have been sugested
tracks_embed = []

# initialize the SpotifyExplorer
spotify_explorer = SpotifyExplorer(numFiles=0, retrainNNC=False)

# Set up Spotipy
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="f2458f8ee1304f43acbb7f6037de86c8",
                                                           client_secret="cd9f1486ebad4d088766160b532d04de"))

CONNECTION_STRING = os.environ['AZURE_STORAGE_CONNECTION_STRING']

blob_service_client = BlobServiceClient.from_connection_string(
    CONNECTION_STRING)


pid_bytes = blob_service_client.get_blob_client(
    "data", "lib/pid.json").download_blob().readall()


pid_dict = json.loads(pid_bytes)
pid = pid_dict['pid']


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

    # Write the playlist to a new JSON file
    playlist_json = json.dumps(playlist, indent=4)

    # Create a blob client using the local file name as the name for the blob if the pid ends in 999, if not read the file with the current pid and append the new playlist to it
    if pid % 1000 == 0:
        blob_service_client.get_blob_client(
            "data", f'data/mpd.slice.{pid}-{pid+999}.json').upload_blob(playlist_json, overwrite=True)
    else:
        blob_client = blob_service_client.get_blob_client(
            "data", f'data/mpd.slice.{pid-(pid % 1000)}-{pid-(pid % 1000)+999}.json')
        blob_data = blob_client.download_blob().readall()
        existing_json = json.loads(blob_data)
        existing_json['playlists'].append(playlist)
        playlist_json = json.dumps(existing_json, indent=4)
        blob_client.upload_blob(playlist_json, overwrite=True)

    # Upload the JSON string to Azure Blob Storage

    # Increment the pid for the next playlist
    pid += 1

    pid_json = json.dumps({'pid': pid})
    blob_service_client.get_blob_client(
        "data", 'lib/pid.json').upload_blob(pid_json, overwrite=True)

    # Make a prediction and get the embed links of the suggested tracks

    tracks_embed = spotify_explorer.predictPlaylist(playlist_series)
    # Redirect to the index page and pass the embed_link to the template
    return redirect(url_for('index', tracks_embed=tracks_embed))


if __name__ == '__main__':
    app.run(host="0.0.0.0")
