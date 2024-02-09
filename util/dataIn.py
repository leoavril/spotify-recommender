import json, display, pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.sparse import dok_matrix
from azure.storage.blob import BlobServiceClient, BlobPrefix
import io


def parseTrackURI(uri):
    return uri.split(":")[2]

def processPlaylistForClustering(playlists, tracks):
    """
    Create sparse matrix mapping playlists to track
    lists that are consumable by most clustering algos
    """

    # List of all track IDs in db
    trackIDs = list(tracks["tid"])
    
    # Map track id to matrix index
    IDtoIDX = {k:v for k,v in zip(trackIDs,range(0,len(trackIDs)))}
    
    playlistIDs = list(playlists["pid"])
    
    print("Create sparse matrix mapping playlists to tracks")
    playlistSongSparse = dok_matrix((len(playlistIDs), len(trackIDs)), dtype=np.float32)

    for i in tqdm(range(len(playlistIDs))):
        # Get playlist and track ids from DF
        playlistID = playlistIDs[i]
    
        if playlistID not in playlists.index:
            continue

        trackID = playlists.loc[playlistID]["tracks"]
        playlistIDX = playlistID
        
        # Get matrix index for track id
        trackIDX = [IDtoIDX.get(i) for i in trackID]
        
        # Set index to 1 if playlist has song
        playlistSongSparse[playlistIDX, trackIDX] = 1 

    return playlistSongSparse.tocsr(), IDtoIDX

def list_blobs_hierarchical(container_client, prefix="data/"):
        files = []
        for blob in container_client.walk_blobs(name_starts_with=prefix, delimiter='/'):
            if isinstance(blob, BlobPrefix):
                files.extend(list_blobs_hierarchical(container_client, prefix=blob.name))
            else:
                files.append(blob)
        return files

def createDFs(idx, numFiles, blob_service_client):
    """
    Creates playlist and track DataFrames from
    json files
    """

    # Get a container client for the 'data' container
    container_client = blob_service_client.get_container_client("data")


    # Get correct number of files to work with
    blobs = list_blobs_hierarchical(container_client)
    files = [blob_service_client.get_blob_client("data", blob.name) for blob in blobs]
    files = files[idx:idx+numFiles]

    tracksSeen = set()
    playlistsLst = []
    trackLst = []

    print("Creating track and playlist DFs")
    for i, file in enumerate(tqdm(files)):

        # Download the blob data
        blob_data = file.download_blob().readall()
        
        # Convert the blob data to a string and load it as json
        data = json.loads(blob_data.decode('utf-8'))
        playlists = data["playlists"]

        # for each playlist
        for playlist in playlists:
            for track in playlist["tracks"]:
                if track["track_uri"] not in tracksSeen:
                    tracksSeen.add(track["track_uri"])
                    trackLst.append(track)
            playlist["tracks"] = [parseTrackURI(x["track_uri"]) for x in playlist["tracks"]]
            playlistsLst.append(playlist)
    
    playlistDF = pd.DataFrame(playlistsLst)

    playlistDF.set_index("pid")

    tracksDF = pd.DataFrame(trackLst)
    # Split id from spotifyURI for brevity
    tracksDF["tid"] = tracksDF.apply(lambda row: parseTrackURI(row["track_uri"]), axis=1)

    playlistClusteredDF, IDtoIDXMap = processPlaylistForClustering(playlists=playlistDF,
                                                       tracks=tracksDF)

    # Add sparseID for easy coercision to sparse matrix for training data
    tracksDF["sparse_id"] = tracksDF.apply(lambda row: IDtoIDXMap[row["tid"]], axis=1)
    tracksDF = tracksDF.set_index("tid")
    
    # Write DFs to blobs
    print(f"Pickling {len(playlistDF)} playlists")
    playlist_pickle = io.BytesIO()
    playlistDF.to_pickle(playlist_pickle)
    playlist_pickle.seek(0)
    blob_service_client.get_blob_client("data", "lib/playlists.pkl").upload_blob(playlist_pickle, overwrite=True)

    print(f"Pickling {len(tracksDF)} tracks")
    tracks_pickle = io.BytesIO()
    tracksDF.to_pickle(tracks_pickle)
    tracks_pickle.seek(0)
    blob_service_client.get_blob_client("data", "lib/tracks.pkl").upload_blob(tracks_pickle, overwrite=True)

    print(f"Pickling clustered playlist")
    playlist_clustered_pickle = io.BytesIO()
    pickle.dump(playlistClusteredDF, playlist_clustered_pickle)
    playlist_clustered_pickle.seek(0)
    blob_service_client.get_blob_client("data", "lib/playlistSparse.pkl").upload_blob(playlist_clustered_pickle, overwrite=True)