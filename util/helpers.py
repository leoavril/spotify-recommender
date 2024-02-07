import random
from scipy.sparse import dok_matrix
import spotipy
import pandas as pd


def playlistToSparseMatrixEntry(playlist, songs):
    """
    Converts a playlist with list of songs
    into a sparse matrix with just one row
    """
    # print(songs.iloc[1:5])
    playlistMtrx = dok_matrix((1, len(songs)))
    tracks = [songs.loc[str(x)]["sparse_id"] for x in list(
        playlist["tracks"]) if str(x) in songs.index]
    playlistMtrx[0, tracks] = 1
    return playlistMtrx.tocsr()


def getPlaylistTracks(playlist, songs):
    return [songs.loc[x] for x in playlist["tracks"]]


def getTrackandArtist(trackURI, songs):
    song = songs.loc[trackURI]
    return (song["track_name"], song["artist_name"])


def obscurePlaylist(playlist, percentToObscure):
    """
    Obscure a portion of a playlist's songs for testing
    """
    k = int(len(playlist['tracks']) * percentToObscure)
    indices = random.sample(range(len(playlist['tracks'])), k)
    obscured = [playlist['tracks'][i] for i in indices]
    tracks = [i for i in playlist['tracks'] +
              obscured if i not in playlist['tracks'] or i not in obscured]
    return tracks, obscured


def parsePlaylistLink(sp, link, pid=100000):

    try:
        # Get playlost details
        playlist = sp.playlist(link)

        # Get tracks
        raw_tracks = playlist['tracks']['items']
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
        playlist_json = {
            'info': {
                'generated_on': '2017-12-04 03:05:11.774401',
                'slice': f'{pid}-{pid+999}',
                'version': 'v1'
            },
            'playlists': [{
                'name': playlist['name'],
                'collaborative': playlist['collaborative'],
                'pid': pid,
                'modified_at': 1468800000,
                'num_tracks': len(tracks),
                'num_albums': len(set([track['album_uri'] for track in tracks])),
                'num_followers': playlist['followers']['total'],
                'tracks': tracks,
                'num_edits': 2,
                'duration_ms': sum([track['duration_ms'] for track in tracks]),
                'num_artists': len(set([track['artist_uri'] for track in tracks]))
            }]
        }

        # create pandas Series object

        playlist_series = pd.Series({
            'name': playlist['name'],
            'collaborative': playlist['collaborative'],
            'pid': pid,
            'modified_at': 1468800000,
            'num_tracks': len(tracks),
            'num_albums': len(set(track['album_uri'] for track in tracks)),
            'num_followers': playlist['followers']['total'],
            'tracks': [track['track_uri'].split(':')[-1] for track in tracks],
            'num_edits': 2,
            'duration_ms': sum(track['duration_ms'] for track in tracks),
            'num_artists': len(set(track['artist_uri'] for track in tracks)),
            'description': None  # replace with actual description if available
        }, name=pid)

        return [playlist_json, playlist_series]

    except spotipy.exceptions.SpotifyException:
        print('Unsupported URL / URI')
    except:
        raise
